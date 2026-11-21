"""Worst-case error analysis CSV and visual diagnostic galleries (matplotlib or PIL fallback)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from wl import CELL, GRID, METHODS, ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]


def _render_error_rgb(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Green=TP, Blue=TN, Red=FP, Yellow=FN at 128x128."""
    rgb = np.zeros((GRID, GRID, 3), dtype=np.uint8)
    tp = (pred == 1) & (gt == 1)
    tn = (pred == 0) & (gt == 0)
    fp = (pred == 1) & (gt == 0)
    fn = (pred == 0) & (gt == 1)
    rgb[tp] = [40, 180, 70]
    rgb[tn] = [30, 40, 80]
    rgb[fp] = [230, 50, 50]
    rgb[fn] = [245, 210, 40]
    return rgb


def _render_gallery_pil(
    records: list[dict[str, Any]],
    probs_map: dict[str, np.ndarray],
    masks_by_name: dict[str, np.ndarray],
    train_images_dir: Path,
    out_path: Path,
    threshold: float = 0.5,
) -> Path:
    """Render 5-column visual gallery using pure PIL so it runs without matplotlib."""
    tile = 128
    pad = 8
    header_h = 24
    cols = 5
    rows_n = len(records)
    canvas = Image.new("RGB", (cols * (tile + pad) + pad, rows_n * (tile + header_h + pad) + pad), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)

    for r_idx, rec in enumerate(records):
        name = rec["image_name"]
        y_off = pad + r_idx * (tile + header_h + pad)
        label = f"{name} | {rec['method']} {rec['kind']} | mIoU={rec['miou']:.3f} gt={rec['gt_frac']:.2f} pr={rec['pred_frac']:.2f}"
        draw.text((pad, y_off + 4), label, fill=(235, 235, 235))

        img_p = train_images_dir / f"{name}.png"
        if img_p.is_file():
            thumb = Image.open(img_p).convert("RGB").resize((tile, tile), Image.Resampling.BILINEAR)
        else:
            thumb = Image.new("RGB", (tile, tile), (60, 60, 60))

        gt = masks_by_name[name]
        pr_prob = probs_map[name]
        pr_bin = (pr_prob >= threshold).astype(np.uint8)

        gt_rgb = Image.fromarray((gt * 255).astype(np.uint8)).convert("RGB")
        prob_rgb = Image.fromarray((np.clip(pr_prob, 0.0, 1.0) * 255).astype(np.uint8)).convert("RGB")
        bin_rgb = Image.fromarray((pr_bin * 255).astype(np.uint8)).convert("RGB")
        err_rgb = Image.fromarray(_render_error_rgb(pr_bin, gt))

        for c_idx, panel in enumerate((thumb, gt_rgb, prob_rgb, bin_rgb, err_rgb)):
            x_off = pad + c_idx * (tile + pad)
            canvas.paste(panel, (x_off, y_off + header_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)
    return out_path


def _render_gallery_mpl(
    records: list[dict[str, Any]],
    probs_map: dict[str, np.ndarray],
    masks_by_name: dict[str, np.ndarray],
    train_images_dir: Path,
    out_path: Path,
    threshold: float = 0.5,
) -> Path:
    n_rows = len(records)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, max(3.0, 2.8 * n_rows)))
    if n_rows == 1:
        axes = axes[None, :]
    col_titles = ["Image", "GT mask128", "Pred prob", "Pred binary", "Error (G=TP, R=FP, Y=FN)"]
    for c_idx, title in enumerate(col_titles):
        axes[0, c_idx].set_title(title, fontsize=10)

    for r_idx, rec in enumerate(records):
        name = rec["image_name"]
        img_p = train_images_dir / f"{name}.png"
        img = (
            np.asarray(Image.open(img_p).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR))
            if img_p.is_file()
            else np.zeros((128, 128, 3), dtype=np.uint8)
        )
        gt = masks_by_name[name]
        pr_prob = probs_map[name]
        pr_bin = (pr_prob >= threshold).astype(np.uint8)
        err = _render_error_rgb(pr_bin, gt)

        axes[r_idx, 0].imshow(img)
        axes[r_idx, 0].set_ylabel(
            f"{name}\n{rec['method']} ({rec['kind']})\nmIoU={rec['miou']:.3f}", fontsize=8
        )
        axes[r_idx, 1].imshow(gt, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r_idx, 2].imshow(pr_prob, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
        axes[r_idx, 3].imshow(pr_bin, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[r_idx, 4].imshow(err, interpolation="nearest")
        for c_idx in range(5):
            axes[r_idx, c_idx].set_xticks([])
            axes[r_idx, c_idx].set_yticks([])

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


def build_gallery(
    probs_path: str | Path,
    data_root: str | Path = ROOT_DIR / "data" / "release",
    out_dir: str | Path | None = None,
    top_k: int = 20,
    threshold: float = 0.5,
) -> tuple[Path, Path]:
    """Write worst_images.csv and worst_gallery.png sorted ascending by per-image mIoU."""
    p_path = Path(probs_path)
    target_dir = Path(out_dir) if out_dir else p_path.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    names, probs = load_probs(p_path)
    meta_by_name, masks_by_name = load_train_ground_truth(data_root)
    gts = np.stack([masks_by_name[n] for n in names], axis=0).astype(np.uint8)
    preds = (probs >= threshold).astype(np.uint8)
    scores = miou_batch(preds, gts)

    records: list[dict[str, Any]] = []
    probs_map: dict[str, np.ndarray] = {}
    for n, pr, sc, gt, pb in zip(names, probs, scores, gts, preds):
        m = meta_by_name[n]
        probs_map[n] = pr
        records.append(
            {
                "image_name": n,
                "method": m["method"],
                "kind": m["kind"],
                "miou": round(float(sc), 6),
                "gt_frac": round(float(gt.mean()), 4),
                "pred_frac": round(float(pb.mean()), 4),
            }
        )

    records.sort(key=lambda r: r["miou"])
    csv_path = target_dir / "worst_images.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_name", "method", "kind", "miou", "gt_frac", "pred_frac"])
        w.writeheader()
        w.writerows(records)

    worst = records[: min(top_k, len(records))]
    train_images_dir = Path(data_root) / "train" / "images"
    png_path = target_dir / "worst_gallery.png"
    if plt is not None:
        _render_gallery_mpl(worst, probs_map, masks_by_name, train_images_dir, png_path, threshold=threshold)
    else:
        _render_gallery_pil(worst, probs_map, masks_by_name, train_images_dir, png_path, threshold=threshold)

    return csv_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Build worst-case image gallery and CSV from val_probs.npz.")
    ap.add_argument("--probs", required=True, help="Path to val_probs.npz")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--out", default=None, help="Output directory (defaults to probs parent)")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    csv_path, png_path = build_gallery(
        probs_path=args.probs,
        data_root=args.data,
        out_dir=args.out,
        top_k=args.top_k,
        threshold=args.threshold,
    )
    print(f"Wrote {csv_path} and {png_path}")


if __name__ == "__main__":
    main()
