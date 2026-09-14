"""Kernel-enforced filesystem sandbox for grading model-generated code (Landlock LSM, ctypes,
stdlib-only). Called INSIDE the short-lived grading child, after chdir(box), BEFORE exec'ing the
program. Unlike the python-level reliability guard (monkeypatched os.remove etc.), this cannot be
bypassed by absolute-path open(), ctypes, importlib.reload(os), or raw syscalls: the kernel denies
every filesystem access outside the allowlist, and the restriction is irreversible for the process
(PR_SET_NO_NEW_PRIVS + Landlock can only ever tighten).

Verified on this cluster 2026-09-01 (login node AND inside the enroot container on a compute
node): kernel 5.15.0-92, LSM list includes landlock, ABI v1, restrict_self works unprivileged,
NFS (/mnt/datafs) paths are governed like any other.

Allowlist model:
  read-only : system dirs (/usr /lib /lib64 /bin /sbin /etc /opt /proc), the python installation
              (sys.prefix / sys.base_prefix — the NFS venv), extra_ro (e.g. multiprocessing's
              temp dir so Manager sockets keep working)
  read-write: the throwaway box dir (+ /dev/null), extra_rw
  everything else (ALL of /mnt/datafs, /home, the repo, ...): NO access, not even read.

Known residual gaps on ABI 1 (kernel 5.15; fixed in later kernels/ABIs, documented here so the
canary tests the right things):
  * truncate(2)/ftruncate on a path is not governed until ABI 3 — but O_WRONLY/O_TRUNC *open* IS
    blocked, so plain open(path, "w") cannot touch anything outside the box; only a raw
    truncate syscall on an existing path slips through (os.truncate is also nulled by the guard).
  * network sockets are not governed until ABI 4 (jobs run with proxies unset anyway).
Fallback: if the kernel lacks Landlock the functions return "unavailable" and the caller keeps
the python-level guard (log it loudly); set MOPD_SANDBOX=require to hard-fail instead,
MOPD_SANDBOX=0 to disable (debug only).
"""
from __future__ import annotations

import ctypes
import os
import resource
import sys
from typing import Iterable, List, Tuple

# x86_64 syscall numbers / constants (linux >= 5.13)
_SYS_landlock_create_ruleset = 444
_SYS_landlock_add_rule = 445
_SYS_landlock_restrict_self = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1
_PR_SET_NO_NEW_PRIVS = 38

# LANDLOCK_ACCESS_FS_* bits (ABI 1 = bits 0..12; TRUNCATE=1<<14 needs ABI 3)
_EXECUTE, _WRITE_FILE, _READ_FILE, _READ_DIR = 1 << 0, 1 << 1, 1 << 2, 1 << 3
_ABI1_ALL = (1 << 13) - 1
_ABI3_TRUNCATE = 1 << 14
_RO = _EXECUTE | _READ_FILE | _READ_DIR
_RW_FILE = _READ_FILE | _WRITE_FILE

SYSTEM_RO = ("/usr", "/lib", "/lib64", "/bin", "/sbin", "/etc", "/opt", "/proc")


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def landlock_abi() -> int:
    """Kernel Landlock ABI version; <= 0 means unavailable."""
    libc = ctypes.CDLL(None, use_errno=True)
    return libc.syscall(_SYS_landlock_create_ruleset, None, 0, _LANDLOCK_CREATE_RULESET_VERSION)


def _python_ro_paths() -> List[str]:
    paths = {sys.prefix, sys.base_prefix, sys.exec_prefix}
    for p in sys.path:
        if p and os.path.isdir(p):
            paths.add(p)
    return sorted(paths)


def restrict_filesystem(rw_paths: Iterable[str], extra_ro: Iterable[str] = ()) -> str:
    """Apply the Landlock allowlist to THIS process (call in the grading child only).
    Returns "landlock:vN" on success, "unavailable:<why>" when the kernel cannot do it
    (caller decides; MOPD_SANDBOX=require raises instead), "disabled" if MOPD_SANDBOX=0."""
    mode = os.environ.get("MOPD_SANDBOX", "1").lower()
    if mode in ("0", "off", "false"):
        return "disabled"
    libc = ctypes.CDLL(None, use_errno=True)
    abi = landlock_abi()
    if abi <= 0:
        msg = f"unavailable:abi={abi},errno={ctypes.get_errno()}"
        if mode == "require":
            raise RuntimeError(f"Landlock required but {msg}")
        return msg
    handled = _ABI1_ALL | (_ABI3_TRUNCATE if abi >= 3 else 0)
    attr = _RulesetAttr(handled)
    fd = libc.syscall(_SYS_landlock_create_ruleset, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        msg = f"unavailable:create,errno={ctypes.get_errno()}"
        if mode == "require":
            raise RuntimeError(msg)
        return msg
    rw_all = handled  # full access beneath rw paths

    def _allow(path: str, access: int) -> None:
        try:
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return
        try:
            pb = _PathBeneath(access, pfd)
            libc.syscall(_SYS_landlock_add_rule, fd, _LANDLOCK_RULE_PATH_BENEATH, ctypes.byref(pb), 0)
        finally:
            os.close(pfd)

    for p in SYSTEM_RO:
        _allow(p, _RO)
    for p in _python_ro_paths():
        _allow(p, _RO)
    for p in extra_ro:
        _allow(p, _RO)
    _allow("/dev/null", _RW_FILE)
    _allow("/dev/urandom", _READ_FILE)
    _allow("/dev/shm", rw_all)  # multiprocessing SemLock/SharedMemory
    for p in rw_paths:
        _allow(p, rw_all)
    if libc.syscall(157, _PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:  # prctl
        os.close(fd)
        msg = f"unavailable:prctl,errno={ctypes.get_errno()}"
        if mode == "require":
            raise RuntimeError(msg)
        return msg
    # on ABI < 3 the kernel does not gate truncate(2); patch that one hole with seccomp
    sc = "seccomp:skip(abi>=3)" if abi >= 3 else seccomp_block_truncate()
    if abi < 3 and mode == "require" and not sc.startswith("seccomp:truncate"):
        os.close(fd)
        raise RuntimeError(f"Landlock ABI {abi} needs the seccomp truncate guard but {sc}")
    r = libc.syscall(_SYS_landlock_restrict_self, fd, 0)
    os.close(fd)
    if r != 0:
        msg = f"unavailable:restrict,errno={ctypes.get_errno()}"
        if mode == "require":
            raise RuntimeError(msg)
        return msg
    return f"landlock:v{abi}+{sc}"


def seccomp_block_truncate() -> str:
    """Deny the path-based ``truncate(2)`` syscall (x86-64 nr 76) with a tiny seccomp-BPF filter.
    On kernel < 6.2 (Landlock ABI < 3) truncate(2) is NOT governed by Landlock, so raw
    ctypes ``libc.truncate(path, 0)`` can zero any DAC-writable file outside the box (verified
    exploitable on this cluster's 5.15 kernel, 2026-09-01). fd-based ftruncate(77) is left alone —
    obtaining an out-of-box fd needs open(), which Landlock denies, and blocking it would break a
    program legitimately truncating its own file. Requires NO_NEW_PRIVS (set by enter_sandbox).
    Returns "seccomp:truncate" / "seccomp:unavailable:<why>" / "seccomp:skip(abi>=3)"."""
    libc = ctypes.CDLL(None, use_errno=True)
    # struct sock_filter { u16 code; u8 jt; u8 jf; u32 k; }
    class Filter(ctypes.Structure):
        _fields_ = [("code", ctypes.c_uint16), ("jt", ctypes.c_uint8),
                    ("jf", ctypes.c_uint8), ("k", ctypes.c_uint32)]

    class Prog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_uint16), ("filter", ctypes.POINTER(Filter))]

    BPF_LD, BPF_W, BPF_ABS, BPF_JMP, BPF_JEQ, BPF_K, BPF_RET = 0x00, 0x00, 0x20, 0x05, 0x10, 0x00, 0x06
    AUDIT_ARCH_X86_64 = 0xC000003E
    SECCOMP_RET_ALLOW = 0x7FFF0000
    SECCOMP_RET_ERRNO = 0x00050000
    EPERM = 1
    NR_OFF, ARCH_OFF = 0, 4            # offsets into struct seccomp_data
    NR_TRUNCATE = 76
    prog = [
        Filter(BPF_LD | BPF_W | BPF_ABS, 0, 0, ARCH_OFF),
        Filter(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, AUDIT_ARCH_X86_64),  # arch==x86_64 -> skip kill
        Filter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),    # foreign arch -> EPERM (fail closed)
        Filter(BPF_LD | BPF_W | BPF_ABS, 0, 0, NR_OFF),
        Filter(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, NR_TRUNCATE),       # nr==truncate -> next is ERRNO
        Filter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ERRNO | EPERM),
        Filter(BPF_RET | BPF_K, 0, 0, SECCOMP_RET_ALLOW),
    ]
    arr = (Filter * len(prog))(*prog)
    bpf = Prog(len(prog), arr)
    PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 22, 2
    r = libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(bpf), 0, 0)
    if r != 0:
        return f"seccomp:unavailable:errno={ctypes.get_errno()}"
    return "seccomp:truncate"


def apply_rlimits(fsize_mb: int = 64, nproc: int | None = None, cpu_extra_s: int | None = None,
                  as_mb: int = 4096) -> None:
    """Belt-and-suspenders resource limits for the grading child. RLIMIT_FSIZE caps any single
    file the program writes. RLIMIT_NPROC stays UNSET by default: it counts threads too, and the
    multiprocessing Queue feeder thread (q.put) would fail under it — process spawning is already
    covered by the nulled subprocess/os.fork guard + Landlock. RLIMIT_AS caps address space:
    without it, generated code allocating unbounded RAM inside its timeout window OOMed whole
    nodes under 32-way concurrent grading (raylet killed by the host OOM-killer -> ray "node
    death": jobs 558243/559366/560997)."""
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (fsize_mb << 20, fsize_mb << 20))
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_AS, (as_mb << 20, as_mb << 20))
    except (ValueError, OSError):
        pass
    if nproc is not None:
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError):
            pass
    if cpu_extra_s is not None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_extra_s, cpu_extra_s + 1))
        except (ValueError, OSError):
            pass


def enter_sandbox(box: str, extra_ro: Iterable[str] = (), extra_rw: Iterable[str] = ()) -> str:
    """chdir is the caller's job; this applies rlimits + Landlock. Returns the landlock status."""
    apply_rlimits()
    import multiprocessing.util as _mpu
    mp_tmp: Tuple[str, ...] = ()
    try:
        # Manager/Queue unix-socket dir must stay reachable or results cannot be reported back
        mp_tmp = (_mpu.get_temp_dir(),)
    except Exception:
        pass
    return restrict_filesystem(rw_paths=(box, *mp_tmp, *extra_rw), extra_ro=extra_ro)
