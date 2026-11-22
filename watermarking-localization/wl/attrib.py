"""Decoder attribution maps with a random-message null (pixel-level evidence from the official extractors).

Why: the remaining loss is the holes of PixelSeal / TrustMark `full` images (small-region IoU 0.16-0.18).
The extractors read the mark through ~32-px receptive fields, so their per-position outputs blur holes.
Their *gradient* does not: d(agreement with the message)/d(pixel) times the pixel's high-frequency content
(input-x-gradient attribution) says which pixels actually carry message evidence. In a hole (clean content)
that contribution has no consistent sign; where the mark is, it is positive.
A-contrario calibration (Three Bricks): the same attribution for K random messages gives a null with the
same content bias, z = (A_true - mean A_rand) / sigma_null, per 4-px cell.

    python -m wl.attrib --split train --device cuda     # prints wm-vs-clean cell AUC per method x kind
    python -m wl.attrib --split test --device cuda
Writes output/attrib/{split}.npz (names, z128 float16), fused with `wl.finalize --attrib`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from wl import CELL, GRID, NBITS, ROOT_DIR, ensure_kit_on_path

OUT_DIR = ROOT_DIR / "output" / "attrib"


def load_extractors(device: str):
    ensure_kit_on_path()
    from wmloc.watermarks import load_watermark

    mdir = ROOT_DIR / "kit" / "models"
    ps = load_watermark("pixelseal", models_dir=mdir, device=device).det.eval()
    tm = load_watermark("trustmark", models_dir=mdir, device=device).net.eval()
    for p in list(ps.parameters()) + list(tm.parameters()):
        p.requires_grad_(False)
    return {"pixelseal": ps, "trustmark": tm}


def per_position_logits(method: str, net, rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable per-position bit logits [B, nbits, h, w] (same paths as W1 in wl.models)."""
    if method == "pixelseal":
        x = net.preprocess(F.interpolate(rgb, size=(256, 256), mode="bilinear", align_corners=False))
        emb = net.pixel_decoder.output_upscaling(net.convnext(x))
        lin = net.pixel_decoder.linear
        out = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
        return out[:, 1:257]
    x = F.interpolate(rgb, size=(245, 245), mode="bilinear", align_corners=False) * 2.0 - 1.0
    h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    for k in ("layer1", "layer2", "layer3", "layer4"):
        h = getattr(net, k)(h)
    return torch.einsum("bchw,oc->bohw", h, net.fc.weight) + net.fc.bias[None, :, None, None]


def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="replicate"), k, stride=1)


def attribution_z(img: np.ndarray, bits: np.ndarray, method: str, net, n_keys: int, seed: int, device: str) -> np.ndarray:
    nb = len(bits)
    rng = np.random.default_rng(seed)
    keys = np.stack([bits] + [rng.integers(0, 2, size=nb) for _ in range(n_keys)]).astype(np.float32) * 2 - 1
    keys_t = torch.from_numpy(keys).to(device)                                   # [K+1, nb]
    rgb = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(device)[None]
    hp = (rgb - _box(rgb, 5)).detach()                                           # high-frequency content
    maps = []
    for k in range(keys_t.shape[0]):
        x = rgb.clone().requires_grad_(True)
        logits = per_position_logits(method, net, x)[:, :nb]
        score = (torch.tanh(logits * 0.5) * keys_t[k][None, :, None, None]).mean()
        (g,) = torch.autograd.grad(score, x)
        a = (g * hp).sum(1, keepdim=True)                                        # input-x-gradient
        maps.append(F.avg_pool2d(a, CELL)[0, 0])                                 # 128x128 cell sums
    m = torch.stack(maps)                                                        # [K+1, 128, 128]
    m = _box(m[:, None], 5)[:, 0]
    null = m[1:]
    mu = null.mean(0)
    sigma = (null - mu).std() + 1e-12
    return torch.nan_to_num((m[0] - mu) / sigma).cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--keys", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()
    data_root = Path(args.data)
    split_dir = data_root / args.split
    meta = json.loads((split_dir / "metadata.json").read_text(encoding="utf-8"))
    if args.limit:
        meta = meta[: args.limit]
    nets = load_extractors(args.device)
    names, zs = [], []
    for i, row in enumerate(meta):
        n, m = row["image_name"], row["method"]
        if m in nets:
            img = np.asarray(Image.open(split_dir / "images" / f"{n}.png").convert("RGB"), dtype=np.uint8)
            bits = np.array([int(c) for c in row["message"]], dtype=np.uint8)[: NBITS[m]]
            seed = int(hashlib.md5(n.encode()).hexdigest()[:8], 16)
            zs.append(attribution_z(img, bits, m, nets[m], args.keys, seed, args.device))
        else:
            zs.append(np.zeros((GRID, GRID), dtype=np.float32))                 # StegaStamp: no extractor here
        names.append(n)
        if (i + 1) % 100 == 0:
            print(f"[attrib {args.split}] {i + 1}/{len(meta)}", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (f"{args.split}.npz" if not args.limit else f"{args.split}_partial.npz")
    np.savez_compressed(out, names=np.array(names), z128=np.stack(zs).astype(np.float16))
    print(f"Wrote {out} ({len(names)} images)")
    if args.split == "train":
        from wl.resynth import _stats
        _stats(names, zs, data_root)


if __name__ == "__main__":
    main()
