"""Generate Cartesian-grid experiment configs and a Slurm job-array manifest."""
from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wl import ROOT_DIR, apply_overrides, load_config  # noqa: E402


def _slug(val_str: str) -> str:
    return (
        val_str.replace("/", "_")
        .replace("[", "")
        .replace("]", "")
        .replace(" ", "")
        .replace(",", "-")
    )


def build_sweep(
    base_config: str | Path,
    grid_args: list[str],
    start_id: int = 110,
    prefix: str = "E",
    configs_out: str | Path = ROOT_DIR / "configs",
    manifest_out: str | Path | None = None,
) -> tuple[list[Path], Path]:
    """Expand `key=v1,v2` grid args into individual YAML configs and a manifest file."""
    axes: list[tuple[str, list[str]]] = []
    for arg in grid_args:
        if "=" not in arg:
            raise ValueError(f"Expected key=v1,v2,... got {arg!r}")
        key, vals_raw = arg.split("=", 1)
        vals = [v.strip() for v in vals_raw.split(",") if v.strip()]
        axes.append((key.strip(), vals))

    base_path = Path(base_config)
    base_cfg = load_config(base_path)
    cfg_dir = Path(configs_out)
    cfg_dir.mkdir(parents=True, exist_ok=True)

    combos = list(itertools.product(*[vals for _, vals in axes])) if axes else [()]
    generated: list[Path] = []
    cur_id = start_id

    for combo in combos:
        overrides = [f"{k}={v}" for (k, _), v in zip(axes, combo)]
        cfg = apply_overrides(base_cfg, overrides)
        tag = "_".join(_slug(f"{k.split('.')[-1]}_{v}") for (k, _), v in zip(axes, combo))
        exp_id = f"{prefix}{cur_id}_{base_path.stem}" + (f"_{tag}" if tag else "")
        cfg["id"] = exp_id
        cfg["parent"] = base_cfg.get("id", base_path.stem)

        out_cfg = cfg_dir / f"{exp_id}.yaml"
        out_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        generated.append(out_cfg)
        cur_id += 1

    man_path = Path(manifest_out) if manifest_out else ROOT_DIR / "output" / "sweeps" / f"{prefix}{start_id}_{base_path.stem}.txt"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text("\n".join(str(p) for p in generated) + "\n", encoding="utf-8")
    return generated, man_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate YAML grid configs and Slurm array manifest.")
    ap.add_argument("base_config", help="Base YAML config (e.g. configs/E101_unet_r34.yaml)")
    ap.add_argument("grid", nargs="+", help="Grid parameters like model.encoder=resnet34,mit_b2")
    ap.add_argument("--start-id", type=int, default=110)
    ap.add_argument("--prefix", default="E")
    ap.add_argument("--configs-out", default=str(ROOT_DIR / "configs"))
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--max-concurrent", type=int, default=4)
    ap.add_argument("--mode", choices=["node", "array"], default="node", help="node: pack 4 GPUs on 1 node via jobs/run_node.sh; array: 1-GPU Slurm job array")
    ap.add_argument("--time", default="04:00:00", help="Walltime for node mode")
    ap.add_argument("--submit", action="store_true", help="Immediately submit via sbatch")
    args = ap.parse_args()

    cfgs, man = build_sweep(
        base_config=args.base_config,
        grid_args=args.grid,
        start_id=args.start_id,
        prefix=args.prefix,
        configs_out=args.configs_out,
        manifest_out=args.manifest,
    )
    print(f"Generated {len(cfgs)} configs; manifest: {man}")
    if args.mode == "node":
        cmd = ["sbatch", f"--job-name=sweep_{args.prefix}{args.start_id}", f"--time={args.time}", "jobs/run_node.sh", str(man)]
    else:
        cmd = ["sbatch", f"--array=0-{len(cfgs) - 1}%{args.max_concurrent}", "jobs/run.sh", str(man)]
    print("Launch command:", " ".join(cmd))
    if args.submit:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
