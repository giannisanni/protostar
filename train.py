"""Trainer: bf16 autocast on cuda, grad accum, WSD schedule, CSV logs, DDP via torchrun.

Single GPU:  python train.py --data /mnt/data/protostar/fineweb10B --steps 1000
Two GPUs:    CUDA_VISIBLE_DEVICES=0,2 torchrun --nproc_per_node=2 train.py ...
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, fields

import torch
import torch.distributed as dist

from data.loader import Shards
from model import Config, Protostar
from muon import SingleDeviceMuonWithAuxAdam


@dataclass
class TrainConfig:
    data_dir: str = ""
    out_dir: str = "out"
    steps: int = 1000
    batch: int = 16  # per-rank micro-batch
    accum: int = 8
    muon_lr: float = 0.02
    adam_lr: float = 3e-4
    warmup: int = 300
    decay_start: int = 0  # 0 = no decay (stable phase); else linear decay to 0 at `steps`
    val_every: int = 250
    val_batches: int = 16
    ckpt_every: int = 1000
    resume: str = ""
    seed: int = 1337


def lr_mult(step, t: TrainConfig):
    if step < t.warmup:
        return (step + 1) / t.warmup
    if t.decay_start and step >= t.decay_start:
        return max(0.0, (t.steps - step) / (t.steps - t.decay_start))
    return 1.0


def build_optimizer(model, t: TrainConfig):
    hidden, rest = [], []
    for name, p in model.named_parameters():
        (hidden if p.ndim >= 2 and "embed" not in name and "lm_head" not in name and "sink" not in name
         else rest).append(p)
    groups = [
        dict(params=hidden, use_muon=True, lr=t.muon_lr, weight_decay=0.01),
        dict(params=rest, use_muon=False, lr=t.adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0),
    ]
    return SingleDeviceMuonWithAuxAdam(groups)


@torch.no_grad()
def val_loss(model, cfg, t, device):
    model.eval()
    mtp_prev, cfg.mtp = cfg.mtp, False  # ponytail: val compares MAIN-head CE only across arms
    loader = Shards(t.data_dir, "val", t.batch, cfg.seq_len)
    losses = []
    for batch, _ in zip(loader.batches(), range(t.val_batches)):
        batch = batch.to(device)
        _, loss = model(batch[:, :-1], targets=batch[:, 1:])
        losses.append(loss.item())
    cfg.mtp = mtp_prev
    model.train()
    return sum(losses) / len(losses)


def train(cfg: Config, t: TrainConfig) -> float:
    ddp = "RANK" in os.environ
    if ddp and not dist.is_initialized():
        dist.init_process_group("nccl")
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.manual_seed(t.seed)

    model = Protostar(cfg).to(device)
    raw = model
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
        from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import bf16_compress_hook

        model.register_comm_hook(dist.group.WORLD, bf16_compress_hook)  # halves comm on the x1 link
    opt = build_optimizer(raw, t)
    loader = Shards(t.data_dir, "train", t.batch, cfg.seq_len, rank=rank, world=world)

    start = 0
    if t.resume:
        # map_location="cpu" + del: loading straight to the GPU keeps a second
        # copy of the optimizer state alive in `ck`, which costs ~0.5GB and
        # OOMs the backward pass on 16GB cards that fit fine in steady state.
        ck = torch.load(t.resume, map_location="cpu", weights_only=True)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        loader.load_state_dict(ck["loader"])
        torch.set_rng_state(ck["rng"].cpu())
        start = ck["step"]
        del ck
        if device.type == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(t.out_dir, exist_ok=True)
    log = open(os.path.join(t.out_dir, "log.csv"), "a") if rank == 0 else None
    if log and start == 0:
        log.write("step,tokens,loss,lr_mult,tok_per_s,val_loss\n")

    def save(step):
        if rank == 0:
            torch.save({"model": raw.state_dict(), "opt": opt.state_dict(),
                        "loader": loader.state_dict(), "rng": torch.get_rng_state(),
                        "step": step, "cfg": vars(cfg)},
                       os.path.join(t.out_dir, f"ckpt_{step:06d}.pt"))

    batches = loader.batches()
    amp = torch.autocast("cuda", torch.bfloat16) if device.type == "cuda" else __import__("contextlib").nullcontext()
    model.train()
    last = 0.0
    t0 = time.time()
    for step in range(start, t.steps):
        for g in opt.param_groups:
            g["lr"] = (t.muon_lr if g.get("use_muon") else t.adam_lr) * lr_mult(step, t)
        total = 0.0
        for micro in range(t.accum):
            batch = next(batches).to(device, non_blocking=True)
            if ddp:  # skip grad sync except on last micro-batch
                model.require_backward_grad_sync = micro == t.accum - 1
            with amp:
                _, loss = model(batch[:, :-1], targets=batch[:, 1:])
            (loss / t.accum).backward()
            total += loss.item() / t.accum
        opt.step()
        opt.zero_grad(set_to_none=True)
        last = total
        if rank == 0 and (step + 1) % 20 == 0:
            toks = 20 * world * t.batch * t.accum * cfg.seq_len
            dt = time.time() - t0
            t0 = time.time()
            vl = ""
            if t.val_every and (step + 1) % t.val_every == 0:
                vl = f"{val_loss(raw, cfg, t, device):.4f}"
            line = f"{step + 1},{(step + 1) * toks // 20},{total:.4f},{lr_mult(step, t):.3f},{toks / dt:.0f},{vl}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
        if t.ckpt_every and (step + 1) % t.ckpt_every == 0:
            save(step + 1)
    save(t.steps)
    if ddp:
        dist.destroy_process_group()
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="", help="JSON file with Config overrides")
    for f in fields(TrainConfig):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = ap.parse_args()
    cfg = Config(**json.load(open(args.config))) if args.config else Config()
    t = TrainConfig(**{f.name: getattr(args, f.name) for f in fields(TrainConfig)})
    train(cfg, t)


if __name__ == "__main__":
    main()
