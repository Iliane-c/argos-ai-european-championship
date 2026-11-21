"""Real & synthetic shard datasets, channel assembly, and fold-aware dataloaders."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from PIL import Image

from wl import CELL, GRID, MAX_BITS, METHOD_TO_ID, ROOT_DIR
from wl.augment import build_train_augment, copy_paste_watermark_safe
from wl.features import FeatureCache, compute_native_view, compute_srm

try:
    import torch
    from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
except ImportError:  # Allow lightweight NumPy inspection without PyTorch installed.
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    ConcatDataset = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    WeightedRandomSampler = None  # type: ignore[assignment]


_CHANNEL_WIDTH = {
    "rgb": 3,
    "srm": 3,
    "native_view": 3,
    "agreement": 1,
    "matched": 1,
}


def count_input_channels(channels: Sequence[str]) -> int:
    """Return total tensor channel depth for a configured list of input channel names."""
    total = 0
    for ch in channels:
        if ch not in _CHANNEL_WIDTH:
            raise ValueError(f"Unknown input channel {ch!r}; valid: {sorted(_CHANNEL_WIDTH)}")
        total += _CHANNEL_WIDTH[ch]
    return total


def load_folds(folds_path: str | Path = ROOT_DIR / "splits" / "folds.json") -> dict[str, int]:
    """Load fixed 5-fold split mapping image_name -> fold_id (0..4)."""
    data = json.loads(Path(folds_path).read_text(encoding="utf-8"))
    fold_of: dict[str, int] = {}
    for k, names in enumerate(data["folds"]):
        for n in names:
            fold_of[str(n)] = k
    return fold_of


def encode_message(
    message: str | np.ndarray | Sequence[int],
    max_bits: int = MAX_BITS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode variable-length binary message into padded float32 (0/1, -1/+1, mask) [max_bits]."""
    if isinstance(message, str):
        raw = np.fromiter((int(c) for c in message.strip()), dtype=np.float32)
    else:
        raw = np.asarray(message, dtype=np.float32).reshape(-1)
    n = min(len(raw), max_bits)
    bits_01 = np.zeros(max_bits, dtype=np.float32)
    bits_signed = np.zeros(max_bits, dtype=np.float32)
    bits_mask = np.zeros(max_bits, dtype=np.float32)
    bits_01[:n] = raw[:n]
    bits_signed[:n] = 2.0 * raw[:n] - 1.0
    bits_mask[:n] = 1.0
    return bits_01, bits_signed, bits_mask


def _upsample_128_to_512(map128: np.ndarray) -> np.ndarray:
    up = np.repeat(np.repeat(np.asarray(map128, dtype=np.float32), CELL, axis=0), CELL, axis=1)
    return up[None, :, :]


def assemble_input_channels(
    img_uint8: np.ndarray,
    method: str,
    channels: Sequence[str] = ("rgb",),
    cached_feats: dict[str, Any] | None = None,
) -> np.ndarray:
    """Build float32 [C, 512, 512] model input stack from requested channel names."""
    parts: list[np.ndarray] = []
    feats = cached_feats or {}
    for ch in channels:
        if ch == "rgb":
            parts.append(np.asarray(img_uint8, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        elif ch == "srm":
            parts.append(compute_srm(img_uint8))
        elif ch == "native_view":
            parts.append(compute_native_view(img_uint8, method))
        elif ch == "agreement":
            arr = feats.get("agreement128")
            if arr is None:
                parts.append(np.full((1, img_uint8.shape[0], img_uint8.shape[1]), 0.5, dtype=np.float32))
            else:
                parts.append(_upsample_128_to_512(arr))
        elif ch == "matched":
            arr = feats.get("matched128")
            if arr is None:
                parts.append(np.full((1, img_uint8.shape[0], img_uint8.shape[1]), 0.5, dtype=np.float32))
            else:
                parts.append(_upsample_128_to_512(arr))
        else:
            raise ValueError(f"Unknown channel {ch!r}")
    return np.concatenate(parts, axis=0).astype(np.float32)


def _format_sample(
    img_uint8: np.ndarray,
    mask128: np.ndarray,
    name: str,
    method: str,
    message: str,
    kind_int: int,
    channels: Sequence[str],
    cached_feats: dict[str, Any] | None,
    as_tensor: bool,
) -> dict[str, Any]:
    stacked = assemble_input_channels(img_uint8, method, channels=channels, cached_feats=cached_feats)
    rgb = np.asarray(img_uint8, dtype=np.float32).transpose(2, 0, 1) / 255.0
    b01, bsigned, bmask = encode_message(message)
    agree128 = (
        np.asarray(cached_feats["agreement128"], dtype=np.float32)
        if cached_feats and "agreement128" in cached_feats
        else np.full((GRID, GRID), 0.5, dtype=np.float32)
    )

    if as_tensor and torch is not None:
        return {
            "image": torch.from_numpy(stacked).float(),
            "rgb": torch.from_numpy(rgb).float(),
            "mask128": torch.from_numpy(mask128.astype(np.float32)),
            "kind": torch.tensor(kind_int, dtype=torch.long),
            "method_id": torch.tensor(METHOD_TO_ID[method], dtype=torch.long),
            "method": method,
            "message_bits": torch.from_numpy(b01).float(),
            "message_signed": torch.from_numpy(bsigned).float(),
            "message_mask": torch.from_numpy(bmask).float(),
            "agreement128": torch.from_numpy(agree128).float(),
            "image_name": name,
        }

    return {
        "image": stacked,
        "rgb": rgb,
        "mask128": mask128.astype(np.float32),
        "kind": kind_int,
        "method_id": METHOD_TO_ID[method],
        "method": method,
        "message_bits": b01,
        "message_signed": bsigned,
        "message_mask": bmask,
        "agreement128": agree128,
        "image_name": name,
    }


class RealDataset(Dataset):
    """Official release train/test dataset with fold filtering, extra channels, and safe augmentations."""

    def __init__(
        self,
        data_root: str | Path = ROOT_DIR / "data" / "release",
        split: str = "train",
        folds: Sequence[int] | None = None,
        folds_path: str | Path = ROOT_DIR / "splits" / "folds.json",
        methods: Sequence[str] | None = None,
        channels: Sequence[str] = ("rgb",),
        cache_dir: str | Path | None = ROOT_DIR / "output" / "cache" / "features",
        augment_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
        copy_paste_prob: float = 0.0,
        as_tensor: bool = True,
        seed: int = 0,
    ):
        self.split_dir = Path(data_root) / split
        self.split = split
        self.channels = tuple(channels)
        self.augment_fn = augment_fn
        self.copy_paste_prob = float(copy_paste_prob)
        self.as_tensor = as_tensor
        self.rng = np.random.default_rng(seed)
        self.cache = FeatureCache(cache_dir) if cache_dir else None

        meta = json.loads((self.split_dir / "metadata.json").read_text(encoding="utf-8"))
        if methods:
            allowed_m = set(methods)
            meta = [m for m in meta if m["method"] in allowed_m]

        self.is_train_split = (self.split_dir / "masks_128.npz").is_file()
        if self.is_train_split and folds is not None:
            fold_of = load_folds(folds_path)
            allowed_f = set(int(f) for f in folds)
            meta = [m for m in meta if fold_of.get(m["image_name"]) in allowed_f]

        self.meta = meta
        self.m128: dict[str, np.ndarray] = {}
        if self.is_train_split:
            with np.load(self.split_dir / "masks_128.npz", allow_pickle=False) as z:
                self.m128 = {str(n): mk.astype(np.uint8) for n, mk in zip(z["names"].tolist(), z["masks"])}

        self.by_method: dict[str, list[int]] = {}
        for idx, row in enumerate(self.meta):
            self.by_method.setdefault(row["method"], []).append(idx)

    def __len__(self) -> int:
        return len(self.meta)

    def _load_raw(self, idx: int) -> dict[str, Any]:
        row = self.meta[idx]
        name = row["image_name"]
        img = np.asarray(Image.open(self.split_dir / "images" / f"{name}.png").convert("RGB"), dtype=np.uint8)
        out: dict[str, Any] = {
            "image": img,
            "image_name": name,
            "method": row["method"],
            "message": row["message"],
        }
        if self.is_train_split:
            out["mask128"] = self.m128[name]
            out["full_watermark"] = bool(row["full_watermark"])
            mask512_p = self.split_dir / "masks" / f"{name}.png"
            if mask512_p.is_file():
                out["mask"] = (np.asarray(Image.open(mask512_p)) > 127).astype(np.uint8)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._load_raw(idx)
        name = item["image_name"]
        method = item["method"]

        if self.is_train_split and self.copy_paste_prob > 0.0 and self.rng.random() < self.copy_paste_prob:
            peers = self.by_method.get(method, [])
            if len(peers) > 1 and "mask" in item:
                other_idx = peers[int(self.rng.integers(0, len(peers)))]
                other = self._load_raw(other_idx)
                if "mask" in other:
                    item = copy_paste_watermark_safe(item, other, self.rng)

        img = item["image"]
        mask128 = item.get("mask128", np.zeros((GRID, GRID), dtype=np.uint8))
        kind_int = (1 if item["full_watermark"] else 0) if self.is_train_split else -1

        if self.augment_fn is not None:
            img, mask128 = self.augment_fn(img, mask128, item, self.rng)

        cached = self.cache.load(self.split, name) if self.cache is not None else None
        return _format_sample(
            img_uint8=img,
            mask128=mask128,
            name=name,
            method=method,
            message=item["message"],
            kind_int=kind_int,
            channels=self.channels,
            cached_feats=cached,
            as_tensor=self.as_tensor,
        )


class SynthShardDataset(Dataset):
    """Dataset reading compressed synthetic .npz shards from `data/synth/<version>/`."""

    def __init__(
        self,
        shard_dir: str | Path,
        methods: Sequence[str] | None = None,
        channels: Sequence[str] = ("rgb",),
        augment_fn: Callable[..., tuple[np.ndarray, np.ndarray]] | None = None,
        as_tensor: bool = True,
        seed: int = 0,
    ):
        self.shard_dir = Path(shard_dir)
        self.channels = tuple(channels)
        self.augment_fn = augment_fn
        self.as_tensor = as_tensor
        self.rng = np.random.default_rng(seed)

        shard_files = sorted(self.shard_dir.glob("*.npz"))
        if not shard_files:
            raise FileNotFoundError(f"No .npz shard files found in {self.shard_dir}")

        allowed = set(methods) if methods else None
        self.shard_paths: list[Path] = shard_files
        self._shard_cache: dict[int, dict[str, np.ndarray]] = {}
        self.index: list[tuple[int, int]] = []
        for s_idx, sp in enumerate(self.shard_paths):
            with np.load(sp, allow_pickle=False) as z:
                m_arr = z["method"].tolist()
            for r_idx, m in enumerate(m_arr):
                if allowed is None or str(m) in allowed:
                    self.index.append((s_idx, r_idx))

    def _get_shard(self, shard_idx: int) -> dict[str, np.ndarray]:
        if shard_idx not in self._shard_cache:
            # Keep memory bounded if there are dozens of large shards.
            if len(self._shard_cache) >= 4:
                self._shard_cache.clear()
            with np.load(self.shard_paths[shard_idx], allow_pickle=False) as z:
                self._shard_cache[shard_idx] = {k: z[k] for k in z.files}
        return self._shard_cache[shard_idx]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_idx, r_idx = self.index[idx]
        shard = self._get_shard(s_idx)
        img = shard["image"][r_idx]
        mask128 = shard["mask128"][r_idx].astype(np.uint8)
        method = str(shard["method"][r_idx])
        message = str(shard["message"][r_idx])
        kind_str = str(shard["kind"][r_idx])
        kind_int = 1 if kind_str == "full" else 0
        name = f"synth_{self.shard_dir.name}_{s_idx:04d}_{r_idx:04d}"

        if self.augment_fn is not None:
            img, mask128 = self.augment_fn(img, mask128, {"method": method, "kind": kind_str}, self.rng)

        cached = None
        if "agreement128" in shard or "matched128" in shard:
            cached = {}
            if "agreement128" in shard:
                cached["agreement128"] = shard["agreement128"][r_idx].astype(np.float32)
            if "matched128" in shard:
                cached["matched128"] = shard["matched128"][r_idx].astype(np.float32)

        return _format_sample(
            img_uint8=img,
            mask128=mask128,
            name=name,
            method=method,
            message=message,
            kind_int=kind_int,
            channels=self.channels,
            cached_feats=cached,
            as_tensor=self.as_tensor,
        )


def _resolve_synth_dir(token: str, synth_root: Path) -> Path:
    name = token[len("synth-") :] if token.startswith("synth-") else token
    cand = synth_root / name
    if cand.is_dir():
        return cand
    direct = Path(token)
    if direct.is_dir():
        return direct
    raise FileNotFoundError(f"Synthetic dataset directory not found for {token!r} (checked {cand})")


def build_dataloaders(
    cfg: dict[str, Any],
    data_root: str | Path = ROOT_DIR / "data" / "release",
    synth_root: str | Path = ROOT_DIR / "data" / "synth",
    folds_path: str | Path = ROOT_DIR / "splits" / "folds.json",
):
    """Construct PyTorch train, validation, and test DataLoaders from an experiment config."""
    if torch is None:
        raise RuntimeError("PyTorch is required to build DataLoaders")

    data_cfg = cfg.get("data", {})
    input_cfg = cfg.get("input", {})
    train_cfg = cfg.get("train", {})
    aug_cfg = cfg.get("augment", {})

    channels = input_cfg.get("channels", ["rgb"])
    methods = data_cfg.get("methods")
    cache_dir = data_cfg.get("cache_dir", str(ROOT_DIR / "output" / "cache" / "features"))
    if cache_dir in ("none", "null", False):
        cache_dir = None

    train_folds = data_cfg.get("train_folds", [1, 2, 3, 4])
    seed = int(train_cfg.get("seed", 0))
    batch_size = int(train_cfg.get("batch", 16))
    num_workers = int(train_cfg.get("workers", 2))
    pin_mem = bool(torch.cuda.is_available())
    aug_fn = build_train_augment(aug_cfg)

    train_sources = data_cfg.get("train", ["real"])
    if isinstance(train_sources, str):
        train_sources = [train_sources]

    real_ds = None
    synth_datasets: list[SynthShardDataset] = []
    for src in train_sources:
        if src == "real":
            real_ds = RealDataset(
                data_root=data_root,
                split="train",
                folds=train_folds,
                folds_path=folds_path,
                methods=methods,
                channels=channels,
                cache_dir=cache_dir,
                augment_fn=aug_fn,
                copy_paste_prob=float(aug_cfg.get("copy_paste_prob", 0.0)),
                as_tensor=True,
                seed=seed,
            )
        else:
            s_dir = _resolve_synth_dir(str(src), Path(synth_root))
            synth_datasets.append(
                SynthShardDataset(
                    shard_dir=s_dir,
                    methods=methods,
                    channels=channels,
                    augment_fn=aug_fn,
                    as_tensor=True,
                    seed=seed,
                )
            )

    parts: list[Dataset] = []
    if real_ds is not None and len(real_ds) > 0:
        parts.append(real_ds)
    parts.extend(sd for sd in synth_datasets if len(sd) > 0)
    if not parts:
        raise ValueError(f"Empty training dataset configuration: {train_sources}")

    if len(parts) == 1:
        train_ds = parts[0]
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem, drop_last=False
        )
    else:
        train_ds = ConcatDataset(parts)
        synth_ratio = float(data_cfg.get("synth_ratio", 0.8))
        n_real = len(real_ds) if real_ds is not None else 0
        n_synth = sum(len(sd) for sd in synth_datasets)
        weights: list[float] = []
        if n_real > 0:
            real_w = (1.0 - synth_ratio) / float(n_real)
            weights.extend([real_w] * n_real)
        if n_synth > 0:
            synth_w = synth_ratio / float(n_synth)
            weights.extend([synth_w] * n_synth)
        epoch_samples = int(data_cfg.get("epoch_samples", len(train_ds)))
        sampler = WeightedRandomSampler(weights, num_samples=epoch_samples, replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin_mem
        )

    val_spec = str(data_cfg.get("val", "f0"))
    if val_spec == "all510":
        val_folds = None
    elif val_spec.startswith("f") and val_spec[1:].isdigit():
        val_folds = [int(val_spec[1:])]
    else:
        raise ValueError(f"Unsupported val split spec: {val_spec!r}")

    val_ds = RealDataset(
        data_root=data_root,
        split="train",
        folds=val_folds,
        folds_path=folds_path,
        methods=methods,
        channels=channels,
        cache_dir=cache_dir,
        augment_fn=None,
        as_tensor=True,
        seed=seed,
    )
    test_ds = RealDataset(
        data_root=data_root,
        split="test",
        folds=None,
        folds_path=folds_path,
        methods=methods,
        channels=channels,
        cache_dir=cache_dir,
        augment_fn=None,
        as_tensor=True,
        seed=seed,
    )

    val_batch = int(train_cfg.get("val_batch", batch_size))
    val_loader = DataLoader(val_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=val_batch, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    return train_loader, val_loader, test_loader
