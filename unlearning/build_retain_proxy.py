"""Pillar 1 Tier A/B retain proxy builder: deduplicated, class-balanced, confident + boundary sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from ul import (
    add_common_io_args,
    extract_features_and_logits,
    infer_vit_dims,
    load_images_uint8,
    load_model,
    resolve_device,
    resolve_task_paths,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _resize_uint8_batch(arr_nhwc: np.ndarray, target_size: int) -> torch.Tensor:
    t = torch.from_numpy(arr_nhwc.copy())
    if t.ndim != 4:
        raise ValueError(f"Expected 4D array in proxy pool, got {tuple(t.shape)}")
    if t.shape[-1] != 3 and t.shape[1] == 3:
        t = t.permute(0, 2, 3, 1).contiguous()
    if t.shape[1] == target_size and t.shape[2] == target_size:
        return t
    x = t.permute(0, 3, 1, 2).float()
    resized = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return resized.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


def _load_pool_images(
    pool_paths: list[Path],
    target_size: int,
    max_candidates: int | None = None,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    collected = 0
    for path in pool_paths:
        if max_candidates is not None and collected >= max_candidates:
            break
        if path.is_dir():
            img_files = sorted(
                p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )
            if not img_files:
                raise ValueError(f"No image files found under pool directory {path}")
            loaded: list[np.ndarray] = []
            for img_file in img_files:
                if max_candidates is not None and collected + len(loaded) >= max_candidates:
                    break
                with Image.open(img_file) as im:
                    fit = ImageOps.fit(
                        im.convert("RGB"), (target_size, target_size), method=Image.Resampling.BICUBIC
                    )
                    loaded.append(np.asarray(fit, dtype=np.uint8))
            if loaded:
                chunk = torch.from_numpy(np.stack(loaded, axis=0))
                chunks.append(chunk)
                collected += len(chunk)
        elif path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            if arr.dtype != np.uint8:
                raise ValueError(f"Pool array {path} must be uint8, got {arr.dtype}")
            if max_candidates is not None:
                arr = arr[: max(0, max_candidates - collected)]
            chunk = _resize_uint8_batch(arr, target_size)
            chunks.append(chunk)
            collected += len(chunk)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                key = "images" if "images" in data.files else data.files[0]
                arr = data[key]
                if max_candidates is not None:
                    arr = arr[: max(0, max_candidates - collected)]
                chunk = _resize_uint8_batch(arr, target_size)
                chunks.append(chunk)
                collected += len(chunk)
        else:
            raise ValueError(f"Unsupported pool path {path}")

    if not chunks:
        raise ValueError("Pool loading yielded zero images")
    return torch.cat(chunks, dim=0)


def _max_cosine_to_forget(
    pool_feats: torch.Tensor,
    forget_feats: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    pf = F.normalize(pool_feats.float(), dim=-1)
    ff = F.normalize(forget_feats.float(), dim=-1)
    max_sims: list[torch.Tensor] = []
    for start in range(0, len(pf), chunk_size):
        sim = pf[start : start + chunk_size] @ ff.T
        max_sims.append(sim.max(dim=1).values)
    return torch.cat(max_sims, dim=0)


def _synthesize_mixup_for_class(
    source_imgs: torch.Tensor,
    source_y: torch.Tensor,
    target_cls: int,
    needed: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    cls_idx = (source_y == target_cls).nonzero(as_tuple=True)[0].numpy()
    other_idx = (source_y != target_cls).nonzero(as_tuple=True)[0].numpy()
    if len(cls_idx) == 0:
        cls_idx = np.arange(len(source_imgs))
    if len(other_idx) == 0:
        other_idx = cls_idx

    synth: list[torch.Tensor] = []
    h, w = int(source_imgs.shape[1]), int(source_imgs.shape[2])
    for i in range(needed):
        a_id = int(rng.choice(cls_idx))
        b_id = int(rng.choice(other_idx))
        img_a = source_imgs[a_id].float()
        img_b = source_imgs[b_id].float()
        if i % 2 == 0:
            # Blend heavily toward target_cls while altering pixel identity completely
            alpha = float(rng.uniform(0.65, 0.85))
            mixed = (alpha * img_a + (1.0 - alpha) * img_b).round().clamp(0, 255).to(torch.uint8)
        else:
            # CutMix patch swap
            mixed = source_imgs[a_id].clone()
            ch = max(2, int(round(h * float(rng.uniform(0.25, 0.45)))))
            cw = max(2, int(round(w * float(rng.uniform(0.25, 0.45)))))
            y0 = int(rng.integers(0, max(1, h - ch + 1)))
            x0 = int(rng.integers(0, max(1, w - cw + 1)))
            mixed[y0 : y0 + ch, x0 : x0 + cw] = source_imgs[b_id, y0 : y0 + ch, x0 : x0 + cw]
        synth.append(mixed)
    return torch.stack(synth, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_io_args(parser)
    parser.add_argument("--pool", type=Path, nargs="+", required=True, help="Pool .npy/.npz or image dir(s)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--min-per-class", type=int, default=10)
    parser.add_argument("--confident-frac", type=float, default=0.70)
    parser.add_argument("--dedup-cos", type=float, default=0.85)
    parser.add_argument("--audit-count", type=int, default=500)
    parser.add_argument("--mixup-fallback", action="store_true")
    parser.add_argument("--max-pool-candidates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not (0.0 <= args.confident_frac <= 1.0):
        raise ValueError(f"--confident-frac must be in [0, 1], got {args.confident_frac}")

    device = resolve_device(args.device)
    task_dir, model_path, forget_path = resolve_task_paths(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    teacher = load_model(model_path, device=device, task_dir=task_dir)
    forget_imgs = load_images_uint8(forget_path)
    num_classes, img_size, _ = infer_vit_dims(teacher, forget_imgs)

    forget_feats, forget_logits = extract_features_and_logits(
        teacher, forget_imgs, device=device, batch_size=args.batch_size
    )
    y_forget = forget_logits.argmax(dim=1)

    pool_imgs = _load_pool_images(args.pool, target_size=img_size, max_candidates=args.max_pool_candidates)
    pool_feats, pool_logits = extract_features_and_logits(
        teacher, pool_imgs, device=device, batch_size=args.batch_size
    )
    max_cos = _max_cosine_to_forget(pool_feats, forget_feats)
    keep_mask = max_cos <= args.dedup_cos
    dedup_dropped = int((~keep_mask).sum().item())

    kept_imgs = pool_imgs[keep_mask]
    kept_logits = pool_logits[keep_mask]
    top2_vals, top2_idx = kept_logits.topk(2, dim=1)
    kept_y = top2_idx[:, 0]
    kept_margin = top2_vals[:, 0] - top2_vals[:, 1]

    selected_chunks: list[torch.Tensor] = []
    per_class_counts: list[int] = []
    synth_added = 0
    rng = np.random.default_rng(args.seed)

    for c in range(num_classes):
        cls_idx = (kept_y == c).nonzero(as_tuple=True)[0]
        if cls_idx.numel() > 0:
            order = cls_idx[torch.argsort(kept_margin[cls_idx], descending=True)]
            n_target = min(args.per_class, len(order))
            n_conf = min(len(order), int(round(n_target * args.confident_frac)))
            conf_pick = order[:n_conf]
            remaining = order[n_conf:]
            n_bound = max(0, n_target - len(conf_pick))
            # Pick boundary/lower-margin examples from the tail of remaining candidates
            bound_pick = remaining[-n_bound:] if n_bound > 0 and len(remaining) > 0 else remaining[:0]
            chosen = torch.cat([conf_pick, bound_pick], dim=0)
            cls_imgs = kept_imgs[chosen]
        else:
            cls_imgs = kept_imgs[:0]

        if args.mixup_fallback and len(cls_imgs) < args.min_per_class:
            needed = args.min_per_class - len(cls_imgs)
            src_imgs = kept_imgs if (kept_y == c).any() else forget_imgs
            src_y = kept_y if (kept_y == c).any() else y_forget
            extra = _synthesize_mixup_for_class(src_imgs, src_y, c, needed, rng)
            cls_imgs = torch.cat([cls_imgs, extra], dim=0) if len(cls_imgs) > 0 else extra
            synth_added += needed

        if len(cls_imgs) > 0:
            selected_chunks.append(cls_imgs)
        per_class_counts.append(int(len(cls_imgs)))

    if not selected_chunks:
        raise ValueError("No retain proxy images survived deduplication and class selection")

    all_selected = torch.cat(selected_chunks, dim=0)
    perm = torch.randperm(len(all_selected), generator=torch.Generator().manual_seed(args.seed))
    all_selected = all_selected[perm]

    if args.audit_count >= len(all_selected):
        raise ValueError(
            f"--audit-count ({args.audit_count}) must be smaller than selected retain total ({len(all_selected)})"
        )

    if args.audit_count > 0:
        retain_fit = all_selected[: -args.audit_count].numpy()
        retain_audit = all_selected[-args.audit_count :].numpy()
    else:
        retain_fit = all_selected.numpy()
        retain_audit = np.empty((0, img_size, img_size, 3), dtype=np.uint8)

    np.save(args.out_dir / "retain_fit.npy", retain_fit)
    np.save(args.out_dir / "retain_audit.npy", retain_audit)

    meta = {
        "pool_total": int(len(pool_imgs)),
        "dedup_cos": args.dedup_cos,
        "dedup_dropped": dedup_dropped,
        "selected_total": int(len(all_selected)),
        "fit_count": int(len(retain_fit)),
        "audit_count": int(len(retain_audit)),
        "synth_mixup_added": synth_added,
        "active_classes": sum(1 for c in per_class_counts if c > 0),
        "min_per_class_actual": min(per_class_counts),
        "max_per_class_actual": max(per_class_counts),
        "per_class_counts": per_class_counts,
    }
    (args.out_dir / "retain_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(
        f"Built retain proxy: pool={len(pool_imgs)} -> kept={len(kept_imgs)} "
        f"(dropped {dedup_dropped} near-dups) -> fit={len(retain_fit)}, audit={len(retain_audit)}, "
        f"active_classes={meta['active_classes']}/{num_classes}"
    )


if __name__ == "__main__":
    main()
