"""Re-predict a trained run from its best checkpoint, with watermark-safe test-time augmentation.

Integer cell shifts (multiples of 4 px) move the image and the evidence together and are undone on the
output, so every view is a valid view of the same watermark; averaging them reduces variance without
any fitting. The shift set is fixed in advance (no tuning on validation).

    python -m wl.infer --run output/E161_w8_noshift --id E161_tta
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from wl import ROOT_DIR
from wl.data import build_dataloaders
from wl.evaluate import evaluate_predictions, format_metrics_summary, load_train_ground_truth, save_probs, write_metrics
from wl.models import build_model
from wl.train import predict_with_tta

DEFAULT_SHIFTS = [[0, 0], [0, 1], [1, 0], [1, 1], [0, -1], [-1, 0], [-1, -1], [1, -1], [-1, 1]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--id", required=True)
    ap.add_argument("--shifts", type=int, default=9, help="Number of shifts from the fixed 3x3 set (1, 4 or 9)")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    args = ap.parse_args()

    run = Path(args.run)
    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cfg.setdefault("data", {})["train"] = ["real"]          # no synth preload: we only predict
    cfg.setdefault("train", {})["workers"] = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, test_loader = build_dataloaders(cfg, data_root=args.data, synth_root=str(ROOT_DIR / "data" / "synth"))
    model = build_model(cfg).to(device)
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    shifts = DEFAULT_SHIFTS[: args.shifts]

    meta, masks = load_train_ground_truth(args.data)
    out = ROOT_DIR / "output" / args.id
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump({**cfg, "id": args.id, "parent": run.name}, sort_keys=False), encoding="utf-8")
    v_names, v_probs = predict_with_tta(model, val_loader, device, shifts=shifts)
    save_probs(out / "val_probs.npz", v_names, v_probs)
    metrics = evaluate_predictions(v_names, v_probs, meta, masks, threshold=0.5)
    metrics.update({"id": args.id, "parent": run.name, "val": cfg.get("data", {}).get("val", "f0"), "data": "tta"})
    write_metrics(out, metrics)
    print(f"[{args.id}] {len(shifts)} shifts | {format_metrics_summary(metrics)}", flush=True)
    t_names, t_probs = predict_with_tta(model, test_loader, device, shifts=shifts)
    save_probs(out / "test_probs.npz", t_names, t_probs)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
