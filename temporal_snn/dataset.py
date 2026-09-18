"""Dataset loading for temporal SNN experiments.

Supports the same CIFAR100 subset convention used in old CSNN experiments:
  "cifar100:0,1,2,3,5,8,13,14,17,19"  → 10 fine-label classes remapped to 0..9

Provides a LatencyEncoder that maps pixel intensities to spike timing:
  bright pixel (≈1.0) → fires at t=0
  dark   pixel (≈0.0) → fires at t=T-1 (or never, if below x_min)
"""
from __future__ import annotations
import os
from typing import Tuple, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


# ─────────────────────────────────────────────────────────────────────────────
# Latency encoder (vectorised, GPU-friendly)
# ─────────────────────────────────────────────────────────────────────────────

class LatencyEncoder:
    """Convert a normalised pixel tensor to a latency-coded spike train.

    Pixels in [0,1].  Bright → early spike, dark → late spike (or silent).

    Input:  [C, H, W]   or   [B, C, H, W]
    Output: [T, C, H, W]  or  [T, B, C, H, W]
    """

    def __init__(self, T: int = 16, x_min: float = 0.1):
        self.T    = T
        self.x_min = x_min

    @torch.no_grad()
    def __call__(self, img: Tensor) -> Tensor:
        """img: float32 [C,H,W] in [0,1]."""
        single = (img.dim() == 3)
        if single:
            img = img.unsqueeze(0)          # [1, C, H, W]
        B, C, H, W = img.shape
        T   = self.T
        dev = img.device

        # Spike time: bright (1.0) → t=0, dark (0.0) → t=T-1
        t_fire = torch.floor((1.0 - img) * (T - 1)).long().clamp(0, T - 1)  # [B,C,H,W]
        mask   = img >= self.x_min                                            # [B,C,H,W]

        spikes = torch.zeros(T, B, C, H, W, device=dev)
        # Scatter: spikes[t_fire[b,c,h,w], b, c, h, w] = 1 where mask
        b_idx = torch.arange(B, device=dev)[:, None, None, None].expand_as(mask)
        c_idx = torch.arange(C, device=dev)[None, :, None, None].expand_as(mask)
        h_idx = torch.arange(H, device=dev)[None, None, :, None].expand_as(mask)
        w_idx = torch.arange(W, device=dev)[None, None, None, :].expand_as(mask)

        spikes[t_fire[mask], b_idx[mask], c_idx[mask], h_idx[mask], w_idx[mask]] = 1.0

        if single:
            spikes = spikes.squeeze(1)      # [T, C, H, W]
        return spikes


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR100 subset dataset
# ─────────────────────────────────────────────────────────────────────────────

def _parse_dataset_spec(spec: str) -> Tuple[str, Optional[List[int]]]:
    """Parse "cifar100:0,1,2,3,5,8,13,14,17,19" → ("cifar100", [0,1,2,3,5,8,13,14,17,19])."""
    if ":" in spec:
        base, rest = spec.split(":", 1)
        labels = [int(x) for x in rest.replace("labels=", "").split(",") if x.strip()]
        return base, labels
    return spec, None


class CIFAR100Subset(Dataset):
    """CIFAR100 filtered to a subset of fine labels, remapped to 0..K-1."""

    def __init__(
        self,
        root: str,
        train: bool,
        fine_labels: Optional[List[int]],
        transform=None,
        download: bool = True,
    ):
        base = datasets.CIFAR100(root=root, train=train, download=download, transform=None)

        if fine_labels is not None:
            label_set = sorted(set(fine_labels))
            remap = {old: new for new, old in enumerate(label_set)}
            idxs  = [i for i, (_, y) in enumerate(base) if y in remap]
            self._data   = [base[i][0] for i in idxs]
            self._labels = [remap[base[i][1]] for i in idxs]
        else:
            self._data   = [base[i][0] for i in range(len(base))]
            self._labels = [base[i][1] for i in range(len(base))]

        self.transform = transform

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        img   = self._data[idx]
        label = self._labels[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_dataloaders(
    dataset_spec: str,
    data_dir: str,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Build train/test DataLoaders for the given dataset spec."""
    base, fine_labels = _parse_dataset_spec(dataset_spec)

    # ── Transforms: [0,1] float32 (latency encoder expects unnormalised) ───
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),   # → [0,1] float32
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
    ])

    if base == "cifar10":
        # CIFAR-10: prefer fast.ai-style ImageFolder (cifar10/train, cifar10/test)
        # to avoid MD5 integrity check on torchvision's pickle format.
        import os as _os
        from torchvision.datasets import ImageFolder as _IF
        _imgfolder_train = _os.path.join(data_dir, "cifar10", "train")
        _imgfolder_test  = _os.path.join(data_dir, "cifar10", "test")
        if _os.path.isdir(_imgfolder_train):
            tr_ds = _IF(_imgfolder_train, transform=train_tf)
            te_ds = _IF(_imgfolder_test,  transform=test_tf)
        else:
            # fallback: standard torchvision pickle (requires MD5-correct data)
            tr_ds = datasets.CIFAR10(root=data_dir, train=True,  download=download, transform=train_tf)
            te_ds = datasets.CIFAR10(root=data_dir, train=False, download=download, transform=test_tf)
    elif "cifar100" in base:
        tr_ds = CIFAR100Subset(data_dir, train=True,  fine_labels=fine_labels, transform=train_tf, download=download)
        te_ds = CIFAR100Subset(data_dir, train=False, fine_labels=fine_labels, transform=test_tf,  download=download)
    else:
        raise ValueError(f"Unknown dataset: {dataset_spec!r}  (supported: 'cifar10', 'cifar100:...')")

    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    te_dl = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=pin_memory)
    return tr_dl, te_dl
