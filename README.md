# protostar

125M-class LLM prototype. Trains on substrate's 2x RTX 5060 Ti (GPU 0 + 2,
`CUDA_VISIBLE_DEVICES=0,2`); the 4060 Ti stays on pulsar serving duty.
Plan and rationale: `docs/plans/2026-07-29-125m-prototype.md`.

- Data dir on substrate: `/mnt/data/protostar` (set `PROTOSTAR_DATA`)
- Preflight: `python env/check_env.py`
- Measured throughput: see table below (Task 4 fills this in)

| setup | tok/s |
|---|---|
| 1x 5060 Ti | TBD |
| 2x 5060 Ti DDP accum=8 | TBD |
