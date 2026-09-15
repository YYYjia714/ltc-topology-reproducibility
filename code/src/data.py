from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class SplitInfo:
    path: Path
    num_samples: int
    history: int
    future: int
    joints: int


class ForecastingDataset(Dataset):
    def __init__(
        self,
        split_file: str | Path,
        normalize: bool = False,
        stats_file: str | Path | None = None,
        max_samples: int = 0,
        seed: int = 42,
    ):
        self.path = Path(split_file)
        if self.path.exists():
            data = np.load(self.path, allow_pickle=True)
            self.inputs = data["inputs"].astype(np.float32)
            self.targets = data["targets"].astype(np.float32)
            self.sequence_ids = data["sequence_ids"]
        else:
            stem = self.path.stem
            parent = self.path.parent
            inputs_path = parent / f"{stem}_inputs.npy"
            targets_path = parent / f"{stem}_targets.npy"
            sequence_ids_path = parent / f"{stem}_sequence_ids.npy"
            if not (inputs_path.exists() and targets_path.exists() and sequence_ids_path.exists()):
                raise FileNotFoundError(f"Split file not found: {self.path}")
            self.inputs = np.load(inputs_path, mmap_mode="r")
            self.targets = np.load(targets_path, mmap_mode="r")
            self.sequence_ids = np.load(sequence_ids_path, mmap_mode="r")
        self.normalize = normalize
        self.mean = None
        self.std = None
        if self.inputs.ndim != 4 or self.targets.ndim != 4:
            raise ValueError(f"Unexpected array shapes in {self.path}")
        if normalize:
            if stats_file is None:
                stats_file = self.path.parent / "normalization_stats.npz"
            stats = np.load(Path(stats_file), allow_pickle=True)
            self.mean = stats["mean"].astype(np.float32)
            self.std = stats["std"].astype(np.float32)
        self.indices = np.arange(self.inputs.shape[0], dtype=np.int64)
        if max_samples > 0 and max_samples < len(self.indices):
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(self.indices, size=max_samples, replace=False))

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        inputs = self.inputs[source_index]
        targets = self.targets[source_index]
        if self.normalize:
            inputs = (inputs - self.mean[0]) / self.std[0]
            targets = (targets - self.mean[0]) / self.std[0]
        x = torch.from_numpy(inputs)
        y = torch.from_numpy(targets)
        return {
            "inputs": x,
            "targets": y,
            "sequence_id": str(self.sequence_ids[source_index]),
        }

    @property
    def info(self) -> SplitInfo:
        return SplitInfo(
            path=self.path,
            num_samples=len(self),
            history=int(self.inputs.shape[1]),
            future=int(self.targets.shape[1]),
            joints=int(self.inputs.shape[2]),
        )
