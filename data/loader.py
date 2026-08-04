"""Memmap shard loader for llm.c-format .bin files (256 int32 header + uint16 tokens).

ponytail: sequential non-overlapping windows, no shuffle beyond shard order —
FineWeb shards are pre-shuffled at document level, which is enough at this scale.
"""

from pathlib import Path

import numpy as np
import torch

MAGIC, VERSION, HDR_BYTES = 20240520, 1, 256 * 4


def write_shard(path, tokens: np.ndarray) -> None:
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype(np.uint16).tobytes())


def read_shard(path) -> np.ndarray:
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == MAGIC, f"{path}: bad magic {header[0]}"
    return np.memmap(path, dtype=np.uint16, mode="r", offset=HDR_BYTES)


class Shards:
    """split: 'train' (all but last shard), 'val' (last shard), 'all'."""

    def __init__(self, dirpath, split, batch, seq_len, rank=0, world=1):
        paths = sorted(Path(dirpath).glob("*.bin"))
        assert paths, f"no .bin shards in {dirpath}"
        if split == "train" and len(paths) > 1:
            paths = paths[:-1]
        elif split == "val":
            paths = paths[-1:]
        self.paths = paths
        self.batch, self.seq_len = batch, seq_len
        self.rank, self.world = rank, world
        self.pos = 0  # global window index, for checkpoint resume

    def state_dict(self):
        return {"pos": self.pos}

    def load_state_dict(self, s):
        self.pos = s["pos"]

    def batches(self):
        window = self.seq_len + 1
        shards = [read_shard(p) for p in self.paths]
        counts = [len(s) // window for s in shards]
        total = sum(counts)
        while True:
            rows = []
            for b in range(self.batch):
                idx = (self.pos * self.world * self.batch + self.rank * self.batch + b) % total
                for shard, n in zip(shards, counts):
                    if idx < n:
                        a = np.asarray(shard[idx * window : (idx + 1) * window], dtype=np.int64)
                        rows.append(torch.from_numpy(a))
                        break
                    idx -= n
            self.pos += 1
            yield torch.stack(rows)


class StatefulShards(Shards):
    """Each lane streams CONTIGUOUS windows: lane b's window at pos+1 continues
    the same token stream as its window at pos. Plain Shards deals consecutive
    windows to DIFFERENT lanes, which makes cross-window memory pure noise —
    the model would (correctly) learn to ignore a persistent state. Lanes get
    disjoint spans of the stream so they never collide."""

    def batches(self):
        window = self.seq_len + 1
        shards = [read_shard(p) for p in self.paths]
        counts = [len(s) // window for s in shards]
        span = sum(counts) // (self.world * self.batch)
        while True:
            rows = []
            for b in range(self.batch):
                idx = (self.rank * self.batch + b) * span + self.pos % span
                for shard, n in zip(shards, counts):
                    if idx < n:
                        a = np.asarray(shard[idx * window : (idx + 1) * window], dtype=np.int64)
                        rows.append(torch.from_numpy(a))
                        break
                    idx -= n
            self.pos += 1
            yield torch.stack(rows)


class CurriculumShards(StatefulShards):
    """StatefulShards with every cur_every-th GLOBAL lane replaced by a
    synthetic retrieval-curriculum stream (accretion/curriculum.py).

    Yields (batch, reset_mask) instead of a bare batch: FineWeb lanes reset
    randomly (reset_prob), curriculum lanes reset exactly at their episode
    boundaries — a random reset mid-episode would grade the model on facts
    its memory no longer holds, teaching it memory CANNOT be trusted.
    Curriculum streams are not resume-exact (fresh episodes after a restart);
    they are synthetic, so nothing is lost."""

    def __init__(self, dirpath, split, batch, seq_len, rank=0, world=1,
                 cur_every=4, reset_prob=0.0625, seed=1337):
        super().__init__(dirpath, split, batch, seq_len, rank=rank, world=world)
        import tiktoken

        from accretion.curriculum import FactStream
        enc = tiktoken.get_encoding("gpt2")
        self.streams = {b: FactStream(enc, seq_len, seed + rank * batch + b)
                        for b in range(batch) if (rank * batch + b) % cur_every == 0}
        self.reset_prob = reset_prob

    def batches(self):
        inner = super().batches()
        while True:
            batch = next(inner)
            mask = torch.rand(self.batch) < self.reset_prob
            for b, fs in self.streams.items():
                toks, reset = fs.next_window()
                batch[b] = torch.tensor(toks, dtype=torch.int64)
                mask[b] = reset
            yield batch, mask
