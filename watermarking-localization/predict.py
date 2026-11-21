"""Convert test_probs.npz into a validated submission.pt and optionally upload to leaderboard."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from wl import GRID, METHODS, ROOT_DIR, load_config
from wl.evaluate import load_probs, load_train_ground_truth
from wl.postproc import apply_postproc, fit_postproc_oof

TASK_ID = "34-watermark-localisation-image"
MAX_SUBMISSION_BYTES = 25 * 1024 * 1024


def submit(file_path: str | Path, base_url: str | None = None, api_key: str | None = None) -> dict[str, Any]:
    """Upload a validated submission.pt file to the hackathon evaluation server."""
    import requests

    url = base_url or os.environ.get("WMLOC_BASE_URL")
    key = api_key or os.environ.get("WMLOC_API_KEY")
    if not url or not key:
        raise ValueError("Set --base-url / WMLOC_BASE_URL and --api-key / WMLOC_API_KEY before submitting.")
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission file not found: {path}")

    with open(path, "rb") as f:
        resp = requests.post(
            f"{url.rstrip('/')}/submit/{TASK_ID}",
            headers={"X-API-Key": key},
            files={"file": (path.name, f, "application/octet-stream")},
            timeout=(10, 120),
        )
    resp.raise_for_status()
    body = resp.json()
    print("Submission response:", body)
    return body


def validate_submission(file_path: str | Path, expected_names: list[str]) -> None:
    """Strictly verify submission.pt matches starter kit schema and size constraints."""
    import torch

    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing submission file: {p}")
    size = p.stat().st_size
    if size > MAX_SUBMISSION_BYTES:
        raise ValueError(f"Submission too large: {size} bytes > {MAX_SUBMISSION_BYTES}")

    payload = torch.load(p, weights_only=True)
    if set(payload.keys()) != {"image_names", "masks"}:
        raise ValueError(f"Unexpected submission keys: {set(payload.keys())}")
    names = payload["image_names"]
    masks = payload["masks"]
    if len(names) != len(expected_names) or set(names) != set(expected_names):
        raise ValueError(f"Image names mismatch: got {len(names)} vs expected {len(expected_names)}")
    if tuple(masks.shape) != (len(expected_names), GRID, GRID):
        raise ValueError(f"Unexpected masks shape: {tuple(masks.shape)}")
    if masks.dtype not in (torch.bool, torch.uint8):
        raise ValueError(f"Unexpected masks dtype: {masks.dtype}")


def make_submission(
    test_probs_path: str | Path,
    out_path: str | Path = ROOT_DIR / "submission.pt",
    data_root: str | Path = ROOT_DIR / "data" / "release",
    val_probs_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> Path:
    """Post-process test probabilities, align to test metadata order, and save validated submission.pt."""
    import torch

    post_cfg: dict[str, Any] = {
        "threshold": 0.5,
        "never_empty": True,
        "min_cells": 24,
        "min_component_cells": 8,
        "rect_snap": False,
    }
    if config_path is not None:
        cfg = load_config(config_path)
        post_cfg.update(cfg.get("post", {}))

    if val_probs_path is None:
        cand = Path(test_probs_path).parent / "val_probs.npz"
        if cand.is_file():
            val_probs_path = cand

    if val_probs_path is not None and post_cfg.get("threshold") == "oof":
        v_names, v_probs = load_probs(val_probs_path)
        meta_by_name, masks_by_name = load_train_ground_truth(data_root)
        fit = fit_postproc_oof(v_names, v_probs, meta_by_name, masks_by_name)
        post_cfg = fit["post_cfg"]
        print(f"OOF postproc fit: raw={fit['raw_score']:.4f} -> oof={fit['oof_score']:.4f}")

    test_meta = json.loads((Path(data_root) / "test" / "metadata.json").read_text(encoding="utf-8"))
    expected_names = [str(r["image_name"]) for r in test_meta]
    t_names, t_probs = load_probs(test_probs_path)
    pos = {n: i for i, n in enumerate(t_names)}
    if set(t_names) != set(expected_names) or len(t_names) != len(expected_names):
        raise ValueError(f"Test probs image set ({len(t_names)}) does not match metadata ({len(expected_names)})")

    order = [pos[n] for n in expected_names]
    aligned_probs = t_probs[order]
    masks = apply_postproc(aligned_probs, test_meta, post_cfg)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"image_names": expected_names, "masks": torch.from_numpy(masks).bool()}, out_file)
    validate_submission(out_file, expected_names)

    # Diagnostic summary per method so empty/degenerate predictions are caught immediately.
    for method in METHODS:
        idx = [i for i, m in enumerate(test_meta) if m["method"] == method]
        sub = masks[idx]
        wm_fracs = sub.mean(axis=(1, 2))
        n_full = int((wm_fracs >= 0.5).sum())
        print(f"  {method:10s}: n={len(idx)} full={n_full} patch={len(idx)-n_full} mean_wm={wm_fracs.mean():.3f}")
    print(f"Validated and wrote {out_file} ({out_file.stat().st_size / 1e6:.2f} MB)")
    return out_file


def main() -> None:
    ap = argparse.ArgumentParser(description="Build and validate submission.pt from test_probs.npz.")
    ap.add_argument("--probs", required=True, help="Path to test_probs.npz")
    ap.add_argument("--val-probs", default=None, help="Optional val_probs.npz for OOF postproc tuning")
    ap.add_argument("--config", default=None, help="Optional YAML config path for post settings")
    ap.add_argument("--data", default=str(ROOT_DIR / "data" / "release"))
    ap.add_argument("--out", default=str(ROOT_DIR / "submission.pt"))
    ap.add_argument("--submit", action="store_true", help="Upload to leaderboard server")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()

    out_file = make_submission(
        test_probs_path=args.probs,
        out_path=args.out,
        data_root=args.data,
        val_probs_path=args.val_probs,
        config_path=args.config,
    )
    if args.submit:
        submit(out_file, base_url=args.base_url, api_key=args.api_key)


if __name__ == "__main__":
    main()
