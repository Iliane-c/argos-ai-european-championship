"""Train-split mask statistics and oracle baselines, per method and per regime (full_watermark flag).

    python analysis/mask_stats.py --data data/release

Prints, for each (method, full_watermark) group: watermarked-area quantiles, how rectangular the mask is
(mask area / bounding-box area), bounding-box aspect ratio, connected components, and the official mIoU of
all-zeros, all-ones, the better of the two, and the ground-truth bounding box used as a prediction.
Needs only numpy and scipy, so it runs without the watermark models.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def miou(pred, gt):
    """Official per-image score (same rule as wmloc.utils.miou_image)."""
    s = 0.0
    for c in (0, 1):
        union = np.logical_or(pred == c, gt == c).sum()
        s += 1.0 if union == 0 else np.logical_and(pred == c, gt == c).sum() / union
    return s / 2


def bbox_mask(g):
    out = np.zeros_like(g)
    ys, xs = np.nonzero(g)
    if len(ys):
        out[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
    return out


def describe(g):
    ys, xs = np.nonzero(g)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    box = bbox_mask(g)
    return dict(
        area=g.mean(),
        fill=g.sum() / (h * w),
        aspect=max(h, w) / min(h, w),
        components=ndimage.label(g)[1],
        has_hole=ndimage.label(box & (1 - g))[1] > 0,
        all0=miou(np.zeros_like(g), g),
        all1=miou(np.ones_like(g), g),
        bbox=miou(box, g),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/release", help="folder that contains train/")
    root = Path(ap.parse_args().data) / "train"
    meta = json.loads((root / "metadata.json").read_text())
    z = np.load(root / "masks_128.npz")
    masks = dict(zip(z["names"].tolist(), z["masks"]))

    rows = [dict(method=m["method"], full=m["full_watermark"], **describe(masks[m["image_name"]].astype(np.uint8)))
            for m in meta]
    quantiles = [0, .1, .25, .5, .75, .9, 1]
    for method in ["pixelseal", "trustmark", "stegastamp", None]:
        for full in [True, False, None]:
            group = [r for r in rows if method in (None, r["method"]) and full in (None, r["full"])]
            col = lambda k: np.array([r[k] for r in group], dtype=float)
            q = lambda k: " ".join(f"{v:.2f}" for v in np.quantile(col(k), quantiles))
            print(f"== {method or 'all'} full_watermark={full} n={len(group)}")
            print(f"  area q{quantiles}: {q('area')}")
            print(f"  fill q: {q('fill')}  rectangles (fill>0.98): {(col('fill') > 0.98).mean():.2f}")
            print(f"  aspect q: {q('aspect')}  components mean: {col('components').mean():.2f}"
                  f"  with hole in bbox: {col('has_hole').mean():.2f}")
            best = np.maximum(col("all0"), col("all1")).mean()
            print(f"  mIoU all0={col('all0').mean():.3f} all1={col('all1').mean():.3f}"
                  f" best-constant-per-image={best:.3f} gt-bbox={col('bbox').mean():.3f}")


if __name__ == "__main__":
    main()
