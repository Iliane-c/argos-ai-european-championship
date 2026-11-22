"""Rank all completed runs by official mIoU and emit terminal or EXPERIMENTS.md tables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from wl import ROOT_DIR


def collect_runs(output_dir: str | Path = ROOT_DIR / "output") -> list[dict[str, Any]]:
    """Read all `output/*/metrics.json` files and sort descending by score."""
    base = Path(output_dir)
    if not base.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    for mfile in sorted(base.glob("*/metrics.json")):
        try:
            data = json.loads(mfile.read_text(encoding="utf-8"))
        except Exception:
            continue
        data.setdefault("id", mfile.parent.name)
        data["_path"] = str(mfile)
        runs.append(data)

    runs.sort(key=lambda r: float(r.get("oof_mean") or r.get("mean") or -1.0), reverse=True)
    return runs


def _fmt(val: Any) -> str:
    return "—" if val is None else f"{float(val):.4f}"


def render_table(runs: list[dict[str, Any]], markdown: bool = False) -> str:
    if not runs:
        return "No metrics.json files found."
    if markdown:
        header = "| ID | Parent | Data | Val | PS | TM | SS | Full | Patch | Mean | OOF |"
        sep = "|---|---|---|---|---|---|---|---|---|---|---|"
        rows = [header, sep]
        for r in runs:
            rows.append(
                f"| {r.get('id', '—')} | {r.get('parent', '—')} | {r.get('data', '—')} | {r.get('val', '—')} | "
                f"{_fmt(r.get('PS'))} | {_fmt(r.get('TM'))} | {_fmt(r.get('SS'))} | "
                f"{_fmt(r.get('full'))} | {_fmt(r.get('patch'))} | {_fmt(r.get('mean'))} | "
                f"{_fmt(r.get('oof_mean'))} |"
            )
        return "\n".join(rows)

    header = f"{'ID':28s} {'Val':6s} {'Mean':>7s} {'OOF':>7s} {'PS':>7s} {'TM':>7s} {'SS':>7s} {'Full':>7s} {'Patch':>7s}"
    lines = [header, "-" * len(header)]
    for r in runs:
        lines.append(
            f"{str(r.get('id', '—'))[:28]:28s} {str(r.get('val', '—'))[:6]:6s} "
            f"{_fmt(r.get('mean')):>7s} {_fmt(r.get('oof_mean')):>7s} "
            f"{_fmt(r.get('PS')):>7s} {_fmt(r.get('TM')):>7s} {_fmt(r.get('SS')):>7s} "
            f"{_fmt(r.get('full')):>7s} {_fmt(r.get('patch')):>7s}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank all experiment runs from output/*/metrics.json.")
    ap.add_argument("--out", default=str(ROOT_DIR / "output"))
    ap.add_argument("--markdown", action="store_true", help="Print Markdown table for EXPERIMENTS.md")
    args = ap.parse_args()
    runs = collect_runs(args.out)
    print(render_table(runs, markdown=args.markdown))


if __name__ == "__main__":
    main()
