"""Decoder-verified localization: pick, per image, the candidate region the decoder can prove.

For every image predicted `patch` (kind from `wl.kind_hybrid` or `wl.refine`), each source run
proposes a candidate region (its main component). Each candidate is then:
1. decoded at the method's native scale (crop of its bounding box, resized by the kit decoder),
2. scored a contrario: agreement of the decoder logits with the *true* message, standardized
   against K random messages on the same logits (Fernandez et al., Three Bricks, WIFS 2023: check
   false-positive rates with random keys). Without the watermark, the score is ~N(0, 1) whatever
   the content, so a z of 4 means p ~ 3e-5;
3. refined by coordinate ascent on the four edges (the dataset embedded each patch at the scale of
   its own rectangle, so the exact rectangle decodes best: E01 / E121).
The candidate with the largest z wins if z >= --z-min (fixed by theory, not tuned); otherwise the
first source's map is kept unchanged. `full` images keep the first source's map.

Nothing is fitted on labels: the reported validation score is honest.

    python -m wl.verify --id E150_verified --sources output/E103_w2_wam_multitask output/E121_analytic_rect_v2 \
        --kind output/E120_kind_const/kind_probs.npz --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from wl import CELL, GRID, MAX_ASPECT, METHODS, METHOD_SHORT, NBITS, ROOT_DIR
from wl.evaluate import evaluate_predictions, format_metrics_summary, load_probs, load_train_ground_truth, save_probs, write_metrics
from wl.features import load_kit_models
from wl.postproc import connected_components

DELTAS = (-8, -4, -2, 2, 4, 8)


class Verifier:
    """A-contrario score of a box (in 128-grid cells): true key vs random keys on the crop's logits."""

    def __init__(self, models: dict[str, Any], n_keys: int, device: str, seed: int = 0):
        self.models, self.device, self.n_keys = models, device, n_keys
        self.rng = np.random.default_rng(seed)

    def set_image(self, img_uint8: np.ndarray, method: str, bits: np.ndarray) -> None:
        import torch

        self.method, self.wm = method, self.models[method]
        self.nbits = min(NBITS[method], len(bits))
        self.base = torch.from_numpy(img_uint8.copy()).permute(2, 0, 1).float().div(255.0)
        true = 2.0 * bits[: self.nbits].astype(np.float32) - 1.0
        rand = self.rng.choice([-1.0, 1.0], size=(self.n_keys, self.nbits)).astype(np.float32)
        self.keys = np.vstack([true[None], rand])  # [K+1, nbits]
        self.cache: dict[tuple[int, int, int, int], float] = {}

    def scores(self, boxes: list[tuple[int, int, int, int]]) -> list[float]:
        import torch

        todo = [b for b in dict.fromkeys(boxes) if b not in self.cache]
        for i in range(0, len(todo), 64):
            chunk = todo[i : i + 64]
            crops = [self.base[:, y0 * CELL : y1 * CELL, x0 * CELL : x1 * CELL] for y0, y1, x0, x1 in chunk]
            res = int(getattr(self.wm, "resolution", 256))
            batch = torch.stack([
                torch.nn.functional.interpolate(c[None], size=(res, res), mode="bilinear", align_corners=False, antialias=True)[0]
                for c in crops
            ])
            with torch.no_grad():
                logits = self.wm.logits(batch).float().cpu().numpy()[:, : self.nbits]
            agree = np.tanh(0.5 * logits) @ self.keys.T / self.nbits  # [B, K+1]
            z = (agree[:, 0] - agree[:, 1:].mean(1)) / (agree[:, 1:].std(1) + 1e-6)
            for b, v in zip(chunk, z):
                self.cache[b] = float(v)
        return [self.cache[b] for b in boxes]

    def refine(self, box: tuple[int, int, int, int], rounds: int = 3) -> tuple[tuple[int, int, int, int], float]:
        best, best_z = box, self.scores([box])[0]
        max_aspect = MAX_ASPECT[self.method]
        for _ in range(rounds):
            cands = []
            for edge in range(4):
                for d in DELTAS:
                    c = list(best)
                    c[edge] += d
                    y0, y1, x0, x1 = c
                    h, w = y1 - y0, x1 - x0
                    if 0 <= y0 < y1 <= GRID and 0 <= x0 < x1 <= GRID and min(h, w) >= 3 and max(h, w) / min(h, w) <= max_aspect:
                        cands.append(tuple(c))
            if not cands:
                break
            zs = self.scores(cands)
            j = int(np.argmax(zs))
            if zs[j] <= best_z + 1e-6:
                break
            best, best_z = cands[j], zs[j]
        return best, best_z


def candidate_box(prob: np.ndarray, thr: float = 0.5) -> tuple[tuple[int, int, int, int], float] | None:
    """Bounding box (y0, y1, x0, x1) of the largest component above thr, and its fill ratio."""
    mask = prob >= thr
    if not mask.any():
        k = 24  # the small part is never empty: take the top cells
        idx = np.argpartition(prob.reshape(-1), -k)[-k:]
        mask = np.zeros(prob.size, dtype=bool)
        mask[idx] = True
        mask = mask.reshape(prob.shape)
    labels, n = connected_components(mask)
    if n == 0:
        return None
    sizes = np.bincount(labels.reshape(-1))[1:]
    comp = labels == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(comp)
    box = (int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1)
    fill = float(comp.sum()) / float((box[1] - box[0]) * (box[3] - box[2]))
    return box, fill


def render(prob: np.ndarray, box: tuple[int, int, int, int], fill: float, rect_fill: float) -> np.ndarray:
    """Rectangle-like candidates become the refined box; object-shaped ones keep their cells inside it."""
    y0, y1, x0, x1 = box
    out = np.zeros_like(prob, dtype=np.float32)
    if fill >= rect_fill:
        out[y0:y1, x0:x1] = 1.0
    else:
        out[y0:y1, x0:x1] = (prob[y0:y1, x0:x1] >= 0.5).astype(np.float32)
        if out.sum() == 0:
            out[y0:y1, x0:x1] = 1.0
    return out


def run_split(split, names, meta, sources, kind, verifier, data_root, z_min, rect_fill, log):
    out_probs, decisions = [], {}
    split_dir = Path(data_root) / split
    for n in names:
        base_map = sources[0][n]
        if kind.get(n, 1.0) >= 0.5:  # full: keep the first source
            out_probs.append(base_map)
            continue
        img = np.asarray(Image.open(split_dir / "images" / f"{n}.png").convert("RGB"), dtype=np.uint8)
        bits = np.array([int(c) for c in meta[n]["message"]], dtype=np.uint8)
        verifier.set_image(img, meta[n]["method"], bits)
        # 1. Verify the primary model's own region as it is (no refinement).
        primary_cb = candidate_box(base_map)
        z0 = verifier.scores([primary_cb[0]])[0] if primary_cb is not None else -np.inf
        if z0 >= z_min:
            out_probs.append(base_map)  # verified: keep the model's map untouched
            decisions[n] = {"source": 0, "z": round(z0, 2), "box": list(primary_cb[0]), "refined": False}
        else:
            # 2. Not verified: refine every source's region (the primary included) and keep the most
            #    significant one, if any reaches z_min.
            cands = []
            for si, src in enumerate(sources):
                cb = primary_cb if si == 0 else candidate_box(src[n])
                if cb is None:
                    continue
                box, z = verifier.refine(cb[0])
                cands.append((z, si, box, cb[1]))
            best = max(cands, default=None, key=lambda c: c[0])
            if best is not None and best[0] >= z_min:
                z, si, box, fill = best
                out_probs.append(render(sources[si][n], box, fill, rect_fill))
                decisions[n] = {"source": si, "z": round(z, 2), "box": list(box), "fill": round(fill, 3), "refined": True}
            else:
                out_probs.append(base_map)
                decisions[n] = {"source": -1, "z": round(float(z0), 2)}
        if len(decisions) % 50 == 0:
            log(f"[verify {split}] {len(decisions)} patch images")
    return np.stack(out_probs).astype(np.float32), decisions


def load_source(run: Path, split: str) -> dict[str, np.ndarray]:
    names, probs = load_probs(run / ("val_probs.npz" if split == "train" else "test_probs.npz"))
    return {str(k): v.astype(np.float32) for k, v in zip(names, probs)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True)
    ap.add_argument("--sources", nargs="+", required=True, help="Run dirs; the first one is the fallback / full-image map")
    ap.add_argument("--kind", required=True, help="kind_probs.npz (wl.kind_hybrid)")
    ap.add_argument("--z-min", type=float, default=4.0, help="Significance needed to override the first source")
    ap.add_argument("--rect-fill", type=float, default=0.8, help="Component fill ratio above which it is drawn as a rectangle")
    ap.add_argument("--keys", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args()

    meta_by_name, masks_by_name = load_train_ground_truth(args.data)
    test_meta = json.loads((Path(args.data) / "test" / "metadata.json").read_text(encoding="utf-8"))
    meta_all = {**meta_by_name, **{r["image_name"]: r for r in test_meta}}
    kz = np.load(args.kind)
    kind = {**{str(n): float(p) for n, p in zip(kz["train_names"], kz["train_p_full_oof"])},
            **{str(n): float(p) for n, p in zip(kz["test_names"], kz["test_p_full"])}}
    verifier = Verifier(load_kit_models(list(METHODS), device=args.device), args.keys, args.device)
    out_dir = Path(args.out) / args.id
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [Path(s) for s in args.sources]

    val_src = [load_source(r, "train") for r in runs]
    v_names = sorted(set.intersection(*[set(s) for s in val_src]))
    v_probs, v_dec = run_split("train", v_names, meta_all, val_src, kind, verifier, args.data, args.z_min, args.rect_fill, print)
    save_probs(out_dir / "val_probs.npz", v_names, v_probs)
    base = evaluate_predictions(v_names, np.stack([val_src[0][n] for n in v_names]), meta_by_name, masks_by_name, threshold=0.5)
    metrics = evaluate_predictions(v_names, v_probs, meta_by_name, masks_by_name, threshold=0.5)
    metrics.update({"id": args.id, "sources": [r.name for r in runs], "z_min": args.z_min, "val": f"intersection-{len(v_names)}",
                    "first_source_same_images": base["mean"]})
    write_metrics(out_dir, metrics)
    print(f"[{args.id}] first source on the same {len(v_names)} images: {base['mean']:.4f} -> verified: {metrics['mean']:.4f}")
    print(f"   {format_metrics_summary(metrics)}")
    kept = sum(d["source"] == 0 and not d.get("refined") for d in v_dec.values())
    fixed = sum(d["source"] >= 0 and d.get("refined", False) for d in v_dec.values())
    none = sum(d["source"] == -1 for d in v_dec.values())
    print(f"   patch images: primary verified as is={kept}, repaired by a verified box={fixed}, "
          f"nothing significant={none} (of {len(v_dec)})")

    decisions = {"train": v_dec}
    if not args.skip_test:
        test_src = [load_source(r, "test") for r in runs]
        t_names = [r["image_name"] for r in test_meta]
        t_probs, t_dec = run_split("test", t_names, meta_all, test_src, kind, verifier, args.data, args.z_min, args.rect_fill, print)
        save_probs(out_dir / "test_probs.npz", t_names, t_probs)
        decisions["test"] = t_dec
    (out_dir / "verify.json").write_text(json.dumps(decisions, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
