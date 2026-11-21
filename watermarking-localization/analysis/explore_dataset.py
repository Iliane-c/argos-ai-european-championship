# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "pillow", "matplotlib"]
# ///
"""Explore the watermark-localization dataset: statistics, a per-image table and figures.

    uv run analysis/explore_dataset.py                      # reads data/release, writes output/explore
    uv run analysis/explore_dataset.py --data DIR --out DIR

Writes to the output folder:
    summary.md                  the statistics tables (also printed)
    train_masks.csv             one row per train image, easy to sort and filter
    fig1_watermarked_fraction.png   how much of each image is watermarked
    fig2_examples_<method>.png      train images with the watermarked area in red
    fig3_mask_gallery.png           many ground-truth masks, to see the shapes
    fig4_where.png                  where the region to find sits in the image
    fig5_region_size.png            size of the region to find, in cells
    fig6_cell_rule.png              pixel mask vs the official 128 x 128 cell grid
    fig7_test_examples.png          test images (no masks)

Terms used:
    cell: one of the 128 x 128 scoring cells (4 x 4 pixels); the official mask is 1 where at least 8 of its
        16 pixels are watermarked.
    kind "full":  full_watermark=True, the whole image was watermarked, then holes were made clean again.
    kind "patch": full_watermark=False, only a patch of a clean image is watermarked.
    target region: the smaller of the two classes, which is the part we must find: the clean holes for
        "full" images, the watermarked patch for "patch" images.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

METHODS = ["pixelseal", "trustmark", "stegastamp"]
KINDS = {True: "full", False: "patch"}
RECT_FILL = 0.98  # a region counts as a rectangle if it fills at least this much of its bounding box


def miou(pred, gt):
    """Official per-image score (same rule as kit/wmloc/utils.py miou_image)."""
    s = 0.0
    for c in (0, 1):
        union = np.logical_or(pred == c, gt == c).sum()
        s += 1.0 if union == 0 else np.logical_and(pred == c, gt == c).sum() / union
    return s / 2


def box_of(region):
    ys, xs = np.nonzero(region)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def describe(name, meta, mask128):
    """One table row for a train image, from its official 128 x 128 mask."""
    full = meta["full_watermark"]
    target = (mask128 == 0) if full else (mask128 == 1)  # the part we must find
    y0, y1, x0, x1 = box_of(target)
    h, w = y1 - y0, x1 - x0
    n_parts = ndimage.label(target)[1]
    fill = target.sum() / (h * w)
    box_pred = np.zeros_like(mask128)
    box_pred[y0:y1, x0:x1] = 1
    if full:
        box_pred = 1 - box_pred  # the box is the hole, the rest is watermarked
    return dict(
        image_name=name, method=meta["method"], kind=KINDS[full],
        watermarked_fraction=round(float(mask128.mean()), 4),
        target_cells=int(target.sum()), target_fraction=round(float(target.mean()), 4),
        target_parts=n_parts,
        target_shape="rectangle" if n_parts == 1 and fill >= RECT_FILL else "irregular",
        target_box_fill=round(float(fill), 3), target_aspect=round(max(h, w) / min(h, w), 2),
        target_box_y0=int(y0), target_box_x0=int(x0), target_box_h=int(h), target_box_w=int(w),
        touches_border=bool(y0 == 0 or x0 == 0 or y1 == 128 or x1 == 128),
        miou_all_zeros=round(miou(np.zeros_like(mask128), mask128), 4),
        miou_all_ones=round(miou(np.ones_like(mask128), mask128), 4),
        miou_true_box=round(miou(box_pred, mask128), 4),
    )


def quant(values):
    q = np.quantile(values, [0.1, 0.5, 0.9])
    return f"{q[1]:.3f} [{q[0]:.3f}–{q[2]:.3f}]"


def summary_tables(rows, test_meta):
    lines = ["# Dataset summary", "",
             "## Sizes", "",
             "| split | pixelseal | trustmark | stegastamp | total |", "|---|---|---|---|---|"]
    tr = [sum(r["method"] == m for r in rows) for m in METHODS]
    te = [sum(t["method"] == m for t in test_meta) for m in METHODS]
    lines += [f"| train | {' | '.join(map(str, tr))} | {sum(tr)} |",
              f"| test | {' | '.join(map(str, te))} | {sum(te)} |", "",
              "Message length: pixelseal 256 bits, trustmark and stegastamp 100 bits. All images are 512 x 512 RGB.",
              "", "## Train images by kind", "",
              "Target region = the part to find (clean holes for `full`, watermarked patch for `patch`).", "",
              "| method | kind | n | watermarked fraction, median [p10–p90] | target fraction, median [p10–p90] "
              "| target is a rectangle | target in 2+ parts | target touches border "
              "| mIoU all-zeros | mIoU all-ones | mIoU true box |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in METHODS + ["all"]:
        for kind in ["full", "patch", "all"]:
            g = [r for r in rows if m in ("all", r["method"]) and kind in ("all", r["kind"])]
            col = lambda k: np.array([r[k] for r in g], dtype=float)
            lines.append(
                f"| {m} | {kind} | {len(g)} | {quant(col('watermarked_fraction'))} | {quant(col('target_fraction'))} "
                f"| {np.mean([r['target_shape'] == 'rectangle' for r in g]):.0%} | {(col('target_parts') > 1).mean():.0%} "
                f"| {col('touches_border').mean():.0%} | {col('miou_all_zeros').mean():.3f} "
                f"| {col('miou_all_ones').mean():.3f} | {col('miou_true_box').mean():.3f} |")
    wf = np.array([r["watermarked_fraction"] for r in rows])
    lines += ["", "## Gap between the two kinds", "",
              f"- `full` images: watermarked fraction from {wf[[r['kind'] == 'full' for r in rows]].min():.2f} "
              f"to {wf[[r['kind'] == 'full' for r in rows]].max():.2f}",
              f"- `patch` images: watermarked fraction from {wf[[r['kind'] == 'patch' for r in rows]].min():.3f} "
              f"to {wf[[r['kind'] == 'patch' for r in rows]].max():.2f}",
              "- No image is between the two ranges."]
    return "\n".join(lines)


def overlay(img, mask512, size=256):
    """Image with the watermarked area tinted red and its border drawn."""
    x = img.astype(np.float32)
    m = mask512.astype(bool)
    x[m] = 0.55 * x[m] + 0.45 * np.array([255, 0, 0])
    edge = m ^ ndimage.binary_erosion(m, iterations=3)
    x[edge] = [255, 255, 0]
    return np.asarray(Image.fromarray(x.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR))


def fig_fraction(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, m in zip(axes, METHODS):
        for kind, color in [("patch", "tab:blue"), ("full", "tab:red")]:
            v = [r["watermarked_fraction"] for r in rows if r["method"] == m and r["kind"] == kind]
            ax.hist(v, bins=bins, color=color, alpha=0.8, label=f"{kind} (n={len(v)})")
        ax.set_title(m)
        ax.set_xlabel("fraction of the image that is watermarked")
        ax.legend()
    axes[0].set_ylabel("number of train images")
    fig.suptitle("Two kinds of images: almost fully watermarked (red) or a small watermarked patch (blue)")
    fig.tight_layout()
    fig.savefig(out / "fig1_watermarked_fraction.png", dpi=110)
    plt.close(fig)


def fig_examples(rows, root, out, rng):
    for m in METHODS:
        fig, axes = plt.subplots(2, 5, figsize=(20, 8.6))
        for row_axes, kind in zip(axes, ["full", "patch"]):
            pick = rng.choice([r for r in rows if r["method"] == m and r["kind"] == kind], 5, replace=False)
            for ax, r in zip(row_axes, pick):
                img = np.asarray(Image.open(root / "images" / f"{r['image_name']}.png").convert("RGB"))
                mask = np.asarray(Image.open(root / "masks" / f"{r['image_name']}.png")) > 127
                ax.imshow(overlay(img, mask))
                ax.set_title(f"{kind} | {r['image_name']}\nwatermarked {r['watermarked_fraction']:.0%}, "
                             f"target {r['target_shape']}", fontsize=9)
                ax.axis("off")
        fig.suptitle(f"{m}: red = watermarked. Top row: kind full (find the clean holes). "
                     f"Bottom row: kind patch (find the red patch).")
        fig.tight_layout()
        fig.savefig(out / f"fig2_examples_{m}.png", dpi=80)
        plt.close(fig)


def fig_gallery(rows, masks, out, rng):
    fig, axes = plt.subplots(6, 12, figsize=(18, 9.5))
    for row_axes, (m, kind) in zip(axes, [(m, k) for m in METHODS for k in ["full", "patch"]]):
        pick = rng.choice([r for r in rows if r["method"] == m and r["kind"] == kind], 12, replace=False)
        for ax, r in zip(row_axes, pick):
            ax.imshow(masks[r["image_name"]], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
        row_axes[0].set_ylabel(f"{m}\n{kind}", fontsize=9)
    fig.suptitle("Ground-truth masks on the 128 x 128 cell grid (white = watermarked)")
    fig.tight_layout()
    fig.savefig(out / "fig3_mask_gallery.png", dpi=80)
    plt.close(fig)


def fig_where(rows, masks, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, kind in zip(axes, ["full", "patch"]):
        g = [r for r in rows if r["kind"] == kind]
        heat = np.mean([(masks[r["image_name"]] == 0) if kind == "full" else masks[r["image_name"]] for r in g], 0)
        im = ax.imshow(heat, cmap="magma", interpolation="nearest")
        ax.set_title(f"{kind}: how often each cell is target\n(" +
                     ("clean hole" if kind == "full" else "watermarked patch") + f", n={len(g)})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "fig4_where.png", dpi=100)
    plt.close(fig)


def fig_size(rows, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    bins = np.logspace(0, np.log10(128 * 128), 40)
    for ax, kind in zip(axes, ["full", "patch"]):
        for shape, color in [("rectangle", "tab:green"), ("irregular", "tab:purple")]:
            v = [r["target_cells"] for r in rows if r["kind"] == kind and r["target_shape"] == shape]
            ax.hist(v, bins=bins, alpha=0.75, color=color, label=f"{shape} (n={len(v)})")
        ax.set_xscale("log")
        ax.set_xlabel("target size in cells (log scale; the image has 16,384 cells)")
        ax.set_title(f"{kind}: size of the " + ("clean holes" if kind == "full" else "watermarked patch"))
        ax.legend()
    axes[0].set_ylabel("number of train images")
    fig.tight_layout()
    fig.savefig(out / "fig5_region_size.png", dpi=100)
    plt.close(fig)


def fig_cell_rule(rows, root, masks, out):
    """Zoom on a region border: pixel mask vs the official cells, plus how much the cell rule changes."""
    r = next(r for r in rows if r["kind"] == "patch" and r["target_shape"] == "irregular" and r["target_cells"] > 400)
    px = np.asarray(Image.open(root / "masks" / f"{r['image_name']}.png")) > 127
    cells = masks[r["image_name"]]
    y0, x0 = r["target_box_y0"] * 4, r["target_box_x0"] * 4
    win = (slice(y0, y0 + 96), slice(x0, x0 + 96))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(px[win], cmap="gray", interpolation="nearest")
    axes[0].set_title("pixel mask (96 x 96 px zoom)")
    axes[1].imshow(np.kron(cells, np.ones((4, 4)))[win], cmap="gray", interpolation="nearest")
    axes[1].set_title("official cells: 1 if >= 8 of 16 pixels watermarked")
    for ax in axes:
        ax.set_xticks(np.arange(-0.5, 96, 4), minor=True); ax.set_yticks(np.arange(-0.5, 96, 4), minor=True)
        ax.grid(which="minor", color="tab:red", linewidth=0.3); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{r['image_name']} ({r['method']}, patch)")
    fig.tight_layout()
    fig.savefig(out / "fig6_cell_rule.png", dpi=100)
    plt.close(fig)


def cell_rule_stats(rows, root):
    """How many cells are mixed (partly watermarked), and the score of a perfect pixel mask after the cell rule."""
    mixed = []
    for r in rows:
        px = (np.asarray(Image.open(root / "masks" / f"{r['image_name']}.png")) > 127).astype(np.float32)
        frac = px.reshape(128, 4, 128, 4).mean(axis=(1, 3))
        mixed.append(((frac > 0) & (frac < 1)).sum() / max(1, r["target_cells"]))
    return float(np.median(mixed))


def fig_test(test_root, test_meta, out, rng):
    fig, axes = plt.subplots(3, 6, figsize=(18, 9.4))
    for row_axes, m in zip(axes, METHODS):
        pick = rng.choice([t for t in test_meta if t["method"] == m], 6, replace=False)
        for ax, t in zip(row_axes, pick):
            ax.imshow(Image.open(test_root / "images" / f"{t['image_name']}.png").convert("RGB").resize((256, 256)))
            ax.set_title(f"{m} | {t['image_name']}", fontsize=9)
            ax.axis("off")
    fig.suptitle("Test images: only image, method and message are given (no kind, no mask)")
    fig.tight_layout()
    fig.savefig(out / "fig7_test_examples.png", dpi=80)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/release", help="folder with train/ and test/")
    ap.add_argument("--out", default="output/explore")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root, test_root, out = Path(a.data) / "train", Path(a.data) / "test", Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    meta = json.loads((root / "metadata.json").read_text())
    test_meta = json.loads((test_root / "metadata.json").read_text())
    z = np.load(root / "masks_128.npz")
    masks = {n: mk.astype(np.uint8) for n, mk in zip(z["names"].tolist(), z["masks"])}
    rows = [describe(m["image_name"], m, masks[m["image_name"]]) for m in meta]

    with open(out / "train_masks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    text = summary_tables(rows, test_meta)
    text += (f"\n\n## Cell rule\n\nMedian number of mixed cells (partly watermarked) per image, relative to the "
             f"target size: {cell_rule_stats(rows, root):.1%}. These border cells are where the 50 % rule decides.\n")
    (out / "summary.md").write_text(text)
    print(text)

    fig_fraction(rows, out)
    fig_examples(rows, root, out, rng)
    fig_gallery(rows, masks, out, rng)
    fig_where(rows, masks, out)
    fig_size(rows, out)
    fig_cell_rule(rows, root, masks, out)
    fig_test(test_root, test_meta, out, rng)
    print(f"\nwrote {out}/summary.md, train_masks.csv and figures")


if __name__ == "__main__":
    main()
