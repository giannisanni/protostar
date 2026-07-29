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
