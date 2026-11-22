"""Concrete example figures for the presentation (CPU, ~1 min): image | ground truth | our masks, on f0 images.

    python -m wl.story_figs --out ~/wl_out/figs

Picks, automatically and from validation only:
  rescued   the image E187 (extra proposals) rescued most: E222 vs E198g
  averaged  the image the two-ensemble average helped most: E224avg vs E222
  success   a StegaStamp patch we get right
  wall      a PixelSeal full image where our final mask has no hole at all
Each figure: the photo, the true mask (blue), then one or two predicted masks (orange), with per-image scores.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from wl import ROOT_DIR
from wl.evaluate import load_probs, load_train_ground_truth, miou_batch
from wl.honest_eval import eiou_decide

BLUE = np.array([36, 87, 214], dtype=np.float32)
ORANGE = np.array([230, 120, 30], dtype=np.float32)


def overlay(img, mask128, color, alpha=0.55):
    m = np.kron(mask128.astype(np.float32), np.ones((4, 4), np.float32))[:, :, None]
    out = img.astype(np.float32) * (1 - alpha * m) + color * alpha * m
    return out.clip(0, 255).astype(np.uint8)


def font(size):
    for p in ("/usr/share/fonts/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def strip(img, panels, path):
    w = 512 * len(panels)
    canvas = Image.new("RGB", (w, 512 + 64), (251, 251, 250))
    d = ImageDraw.Draw(canvas)
    f = font(30)
    for i, (title, arr) in enumerate(panels):
        canvas.paste(Image.fromarray(arr), (512 * i, 64))
        d.text((512 * i + 12, 14), title, fill=(29, 40, 48), font=f)
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/wl_out/figs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    meta, masks = load_train_ground_truth(ROOT_DIR / "data" / "release")
    runs = {"E198g": "output/E198g_final_gate", "E222": "output/E222_e198_multiprop",
            "E224": os.path.expanduser("~/wl_out/E224avg")}
    P = {k: dict(zip(*load_probs(f"{v}/val_probs.npz"))) for k, v in runs.items()}
    names = [n for n in P["E224"] if all(n in P[k] for k in P)]
    M = {k: {n: eiou_decide(P[k][n].astype(np.float32)) for n in names} for k in P}
    S = {k: {n: miou_batch(M[k][n][None], masks[n][None])[0] for n in names} for k in P}
    img = lambda n: np.asarray(Image.open(ROOT_DIR / "data/release/train/images" / f"{n}.png").convert("RGB"))
    meth = lambda n: meta[n]["method"]; kind = lambda n: meta[n]["kind"]
    picks = {
        "rescued": max(names, key=lambda n: S["E222"][n] - S["E198g"][n]),
        "averaged": max(names, key=lambda n: S["E224"][n] - S["E222"][n]),
        "success": max([n for n in names if meth(n) == "stegastamp" and kind(n) == "patch"], key=lambda n: S["E224"][n]),
        "wall": min([n for n in names if meth(n) == "pixelseal" and kind(n) == "full"], key=lambda n: S["E224"][n]),
    }
    for tag, n in picks.items():
        im = img(n)
        truth = overlay(im, masks[n] if kind(n) == "patch" else 1 - masks[n], BLUE)
        if tag == "rescued":
            a, b = "E198g", "E222"
        elif tag == "averaged":
            a, b = "E222", "E224"
        else:
            a, b = None, "E224"
        panels = [(f"{meth(n)}, {kind(n)}", im), ("truth", truth)]
        for k in (a, b):
            if k is None:
                continue
            mk = M[k][n] if kind(n) == "patch" else 1 - M[k][n]
            panels.append((f"{k}: {S[k][n]:.2f}", overlay(im, mk, ORANGE)))
        p = os.path.join(args.out, f"{tag}.png")
        strip(im, panels, p)
        print(tag, n, meth(n), kind(n), {k: round(float(S[k][n]), 3) for k in P}, "->", p)


if __name__ == "__main__":
    main()
