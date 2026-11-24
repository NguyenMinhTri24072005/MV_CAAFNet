"""Dataset & DataLoader đọc `<split>.h5` — SPECS §5.6.

`H5SegDataset` nạp toàn bộ `image`, `mask`, `dist`, `patch_id`, `patient_id` vào RAM một lần.
Với `augment=True` (chỉ tập train), mỗi mẫu được áp một phép thuộc nhóm D4 (xoay 90°·k × lật ngang)
**đồng bộ** lên `image`, `mask`, `dist`. Mọi phép D4 là đẳng cự lưới vuông nên `dist == EDT(mask)`
vẫn đúng tuyệt đối sau biến đổi.

Phép biến đổi của mẫu `i` ở epoch `e` là hàm tất định của `(seed, e, i)`: gọi `set_epoch(e)` trước mỗi
epoch (không dùng `persistent_workers=True`, vì khi đó worker giữ bản sao cũ của dataset).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # cho phép import (và dùng d4_transform) ở máy không có torch
    torch = None
    DataLoader = None
    Dataset = object

SPLITS = ("train", "val", "test")
N_D4 = 8


def d4_transform(arr: np.ndarray, op: int) -> np.ndarray:
    """Phép thứ `op` ∈ [0, 8) của nhóm D4 trên hai trục cuối: xoay `op % 4` lần 90°, lật ngang nếu `op >= 4`."""
    if not 0 <= int(op) < N_D4:
        raise ValueError(f"op phải trong [0, 8), nhận {op}")
    out = np.rot90(arr, k=int(op) % 4, axes=(-2, -1))
    if op >= 4:
        out = out[..., ::-1]
    return np.ascontiguousarray(out)


class H5SegDataset(Dataset):
    """Đọc một tệp `<split>.h5` vào RAM; trả dict tensor float32 `(1, 64, 64)` + `patch_id`, `patient_id`."""

    def __init__(self, h5_path: str | Path, augment: bool = False, seed: int = 0):
        if torch is None:
            raise ImportError("H5SegDataset cần PyTorch")
        import h5py

        self.h5_path = str(h5_path)
        with h5py.File(self.h5_path, "r") as h5:
            self.image = np.asarray(h5["image"][...], dtype=np.float32)
            self.mask = np.asarray(h5["mask"][...], dtype=np.float32)
            self.dist = np.asarray(h5["dist"][...], dtype=np.float32)
            self.patch_id = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h5["patch_id"][...]]
            self.patient_id = [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in h5["patient_id"][...]]
            self.attrs = dict(h5.attrs)
        if not (len(self.image) == len(self.mask) == len(self.dist) == len(self.patch_id)):
            raise ValueError(f"{self.h5_path}: các dataset trong tệp lệch số phần tử")
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.patch_id)

    def op_for(self, i: int) -> int:
        if not self.augment:
            return 0
        return int(np.random.default_rng([self.seed, self.epoch, int(i)]).integers(N_D4))

    def __getitem__(self, i: int) -> dict:
        op = self.op_for(i)
        image, mask, dist = self.image[i], self.mask[i], self.dist[i]
        if op:
            image, mask, dist = d4_transform(image, op), d4_transform(mask, op), d4_transform(dist, op)
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image)),
            "mask": torch.from_numpy(np.ascontiguousarray(mask)),
            "dist": torch.from_numpy(np.ascontiguousarray(dist)),
            "patch_id": self.patch_id[i],
            "patient_id": self.patient_id[i],
        }


def make_loaders(cfg: dict, data_dir: str | Path, dataset: str) -> dict:
    """DataLoader cho `train` (shuffle, augment theo `data.augment`), `val`, `test` (không shuffle)."""
    if torch is None:
        raise ImportError("make_loaders cần PyTorch")
    root = Path(data_dir) / dataset
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["data"].get("num_workers", 0))
    seed = int(cfg.get("seed", 0))
    loaders = {}
    for split in SPLITS:
        ds = H5SegDataset(root / f"{split}.h5", augment=(split == "train" and bool(cfg["data"].get("augment"))), seed=seed)
        generator = torch.Generator()
        generator.manual_seed(seed)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split == "train" else None,
        )
    return loaders
