# mopd_domains — medical / law / finance domain extension (2026-09-07)

New-domain extension for the MOPD pipeline (thinking models, Qwen3-4B-OT3 lineage).
- CLEAN ROOM: implemented WITHOUT reading any mtm code (user requirement).
- Existing repos (mopd, mopd_rl, mopd_poc, mtm) are never modified from here.
- Graders port OFFICIAL benchmark scoring code (sources documented per grader).
- Layout mirrors mopd conventions: graders/, data/ (prep), eval/ (bench loaders),
  tests/ (gold-answer + canary tests required before any large run).
- Large datasets land in ../data/<name> (shared data tree), not in this repo.
