"""Substrate preflight: run before any GPU work. Exits nonzero with a reason."""
import os
import shutil

import torch

assert torch.cuda.is_available(), "CPU-only torch — install cu130 wheel"
n = torch.cuda.device_count()
assert n >= 2, f"need both 5060 Tis visible, found {n} device(s)"
for i in range(n):
    cap = torch.cuda.get_device_capability(i)
    name = torch.cuda.get_device_name(i)
    free, total = torch.cuda.mem_get_info(i)
    print(f"cuda:{i} {name} sm_{cap[0]}{cap[1]} free {free / 1e9:.1f}/{total / 1e9:.1f} GB")

data_dir = os.environ.get("PROTOSTAR_DATA", ".")
free_gb = shutil.disk_usage(data_dir).free / 1e9
assert free_gb > 60, f"need 60GB free at {data_dir} for data+ckpts, have {free_gb:.0f}"

import fla  # noqa: F401  # flash-linear-attention importable

print("env OK")
