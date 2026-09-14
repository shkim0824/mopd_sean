"""CPU tests for the MOPD pieces that need no GPU: placement, teacher-response parsing,
PG advantages, top-k loss (incl. the minimiser property), config/chat helpers."""
import json, math, os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
import torch
import torch.nn.functional as F

from mopd.distill import loss as L
from mopd.distill.placement import parse_teachers, plan
from mopd.distill.teacher_client import TeacherPool
from mopd.common.config import load_config
from mopd.common import chat as C


def test_placement():
    t = parse_teachers("math:/m,code:/c,if:/i", "math=0.35,code=0.30,if=0.35")
    p = plan(["n0", "n1"], t)
    reps = {d: len(u) for d, u in p["endpoints"].items()}
    assert reps == {"math": 3, "code": 2, "if": 3}, reps
    assert p["unused_gpus"] == 0 and len(p["servers"]) == 8
    assert all(s["node"] == "n1" for s in p["servers"]) and p["student"]["node"] == "n0"
    ports = [s["port"] for s in p["servers"]]
    assert len(set(ports)) == 8
    # 6 teachers
    t6 = parse_teachers(",".join(f"d{i}:/p{i}" for i in range(6)), "d0=0.3,d1=0.2,d2=0.125,d3=0.125,d4=0.125,d5=0.125")
    p6 = plan(["n0", "n1"], t6)
    reps6 = {d: len(u) for d, u in p6["endpoints"].items()}
    assert reps6 == {"d0": 2, "d1": 2, "d2": 1, "d3": 1, "d4": 1, "d5": 1}, reps6
    # tp=2 teachers on 2 teacher nodes
    t3 = parse_teachers("math:/m:2,code:/c:2,if:/i:2")
    p3 = plan(["n0", "n1", "n2"], t3)
    assert all(len(s["gpus"]) == 2 for s in p3["servers"]) and len(p3["servers"]) == 8
    assert all(s["gpus"] in ([0, 1], [2, 3], [4, 5], [6, 7]) for s in p3["servers"])
    # too many
    try:
        plan(["n0", "n1"], parse_teachers("a:/a:4,b:/b:4,c:/c:4"))
        assert False
    except AssertionError:
        pass
    print("placement ok")


def test_teacher_parse():
    # vLLM prompt_logprobs shape: [None, {tid: {...}, ...}, ...]
    ids = [11, 22, 33, 44]
    pl = [None,
          {"22": {"logprob": -0.5, "rank": 1}, "7": {"logprob": -1.5, "rank": 2}},
          {"9": {"logprob": -0.1, "rank": 1}, "33": {"logprob": -3.0, "rank": 5}, "8": {"logprob": -0.9, "rank": 2}},
          {"44": {"logprob": -0.2, "rank": 1}, "5": {"logprob": -2.0, "rank": 2}}]
    lp, tk = TeacherPool._parse(pl, ids, start=2, top_k=2)
    assert lp == [-3.0, -0.2]
    assert tk[0] == [[9, 8], [44, 5]], tk[0]
    assert tk[1] == [[-0.1, -0.9], [-0.2, -2.0]]
    lp0, tk0 = TeacherPool._parse(pl, ids, start=0, top_k=0)
    assert lp0 == [0.0, -0.5, -3.0, -0.2] and tk0 is None
    # missing actual token -> very negative
    lp2, _ = TeacherPool._parse([None, {"1": {"logprob": -0.1}}], [5, 6], 1, 0)
    assert lp2[0] < -1000
    print("teacher parse ok")


def test_pg_adv():
    t = torch.tensor([[-1.0, -2.0, -9.0]])
    s = torch.tensor([[-1.5, -1.0, -1.0]])
    m = torch.tensor([[1.0, 1.0, 0.0]])
    a = L.pg_advantages(t, s, m, a_max=5.0)
    assert torch.allclose(a, torch.tensor([[0.5, -1.0, 0.0]])), a
    a2 = L.pg_advantages(t, s, torch.ones(1, 3), a_max=5.0)
    assert a2[0, 2].item() == -5.0
    print("pg adv ok")


def test_topk_loss():
    torch.manual_seed(0)
    V, K, T = 50, 8, 5
    teacher_logits = torch.randn(1, T, V) * 3
    tea_logp_full = F.log_softmax(teacher_logits, -1)
    top = tea_logp_full.topk(K, -1)
    ids, lps = top.indices, top.values
    mask = torch.ones(1, T)
    # (1) at student == teacher the loss is 0
    l0, _ = L.topk_loss_from_logits(teacher_logits.clone(), ids, lps, mask)
    assert abs(l0.item()) < 1e-5, l0
    # (2) loss is minimised at student == teacher: gradient descent from random init decreases it & converges there
    stu = torch.nn.Parameter(torch.randn(1, T, V))
    opt = torch.optim.Adam([stu], lr=0.1)
    prev = None
    for _ in range(300):
        opt.zero_grad()
        l, _ = L.topk_loss_from_logits(stu, ids, lps, mask)
        l.mean().backward()
        opt.step()
    lend, _ = L.topk_loss_from_logits(stu.detach(), ids, lps, mask)
    assert lend.item() < 0.05, lend
    stu_p = F.softmax(stu.detach(), -1).gather(-1, ids)
    assert torch.allclose(stu_p, lps.exp(), atol=0.02), (stu_p - lps.exp()).abs().max()
    # (3) padding ids (-1) and masked positions contribute nothing
    ids2 = ids.clone(); ids2[0, 0, -1] = -1
    m2 = mask.clone(); m2[0, -1] = 0
    la, _ = L.topk_loss_from_logits(teacher_logits, ids2, lps, m2)
    assert abs(la.item()) < 1e-5
    # (4) non-negativity on random students (generalised KL is >= 0 termwise-summed on the support)
    for _ in range(5):
        lr, _ = L.topk_loss_from_logits(torch.randn(1, T, V) * 2, ids, lps, mask)
        assert lr.item() > -1e-6
    print("topk loss ok")


def test_config_and_chat():
    cfg = load_config(None, ["train.lr=5e-7", "data.ratio.math=0.5", "flag=true"],
                      base={"train": {"lr": 1e-6}, "data": {"ratio": {"math": 0.35}}, "flag": False})
    assert cfg.train.lr == 5e-7 and cfg.data.ratio.math == 0.5 and cfg.flag is True
    assert C.normalize_assistant("<think>\nabc\n</think>\n\nans") == "<think>\nabc\n</think>\n\nans"
    assert C.normalize_assistant("ans", "why") == "<think>\nwhy\n</think>\n\nans"
    assert C.normalize_assistant("<think>x</think>ans", thinking=False) == "ans"
    s = C.split_thinking("<think>\nr\n</think>\n\nA")
    assert s == {"reasoning": "r", "answer": "A", "finished_thinking": True}
    assert C.split_thinking("<think>partial")["finished_thinking"] is False
    print("config/chat ok")


def test_qwen3_template_contract():
    """Uses the Qwen3-4B-Base tokenizer files if present (tokenizer only, no weights)."""
    path = os.environ.get("QWEN3_TOKENIZER", "")
    if not path or not os.path.exists(os.path.join(path, "tokenizer_config.json")):
        print("skip template test (set QWEN3_TOKENIZER=<dir with Qwen3 tokenizer>)"); return
    from transformers import AutoTokenizer
    tok = C.prepare_tokenizer(AutoTokenizer.from_pretrained(path))
    assert tok.eos_token == "<|im_end|>" and tok.pad_token == "<|endoftext|>"
    msgs = [{"role": "user", "content": "hi"}]
    p = C.render_prompt(tok, msgs, thinking=True)
    assert p.endswith("<|im_start|>assistant\n"), repr(p[-40:])
    p2 = C.render_prompt(tok, msgs, thinking=False)
    assert p2.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(p2[-60:])
    ids = C.render_prompt_ids(tok, msgs)
    assert tok.decode(ids) == p
    assert C.eos_token_ids(tok) == [151645, 151643]
    print("qwen3 template ok")


if __name__ == "__main__":
    test_placement(); test_teacher_parse(); test_pg_adv(); test_topk_loss(); test_config_and_chat(); test_qwen3_template_contract()
    print("ALL OK")
