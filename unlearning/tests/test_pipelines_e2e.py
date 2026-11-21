"""End-to-end subprocess tests for every Task 3 CLI pipeline and Slurm job script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from tests.conftest import UNLEARNING_DIR


def _run_cli(script_name: str, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["TMPDIR"] = "/tmp"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{UNLEARNING_DIR}:{existing_pp}" if existing_pp else str(UNLEARNING_DIR)
    cmd = [sys.executable, str(UNLEARNING_DIR / script_name), *args]
    res = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise AssertionError(
            f"Command failed ({res.returncode}): {' '.join(cmd)}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
    return res


def _assert_valid_checkpoint(ckpt_path: Path, ref_path: Path, task_dir: Path) -> None:
    assert ckpt_path.is_file()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ref = torch.load(ref_path, map_location="cpu", weights_only=True)
    assert set(sd.keys()) == set(ref.keys())
    for k, v in sd.items():
        assert v.dtype == torch.float32
        assert torch.isfinite(v).all()
        assert v.shape == ref[k].shape

    sys.path.insert(0, str(task_dir))
    import importlib

    tt = importlib.import_module("task_template")
    m = tt.build_model()
    m.load_state_dict(sd, strict=True)
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, tt.IMAGE_SIZE, tt.IMAGE_SIZE))
    assert torch.isfinite(out).all()


def test_full_pipeline_end_to_end(synthetic_task_env: dict[str, Path], tmp_path: Path) -> None:
    task_dir = synthetic_task_env["task_dir"]
    model_path = synthetic_task_env["model_path"]
    pool_path = synthetic_task_env["pool_path"]
    work = tmp_path / "pipeline_out"
    work.mkdir(parents=True, exist_ok=True)

    # 1. diagnose.py
    diag_dir = work / "diagnose"
    _run_cli(
        "diagnose.py",
        ["--task-dir", str(task_dir), "--out-dir", str(diag_dir), "--batch-size", "8", "--save-grids", "4", "--device", "cpu"],
        cwd=tmp_path,
    )
    diag = json.loads((diag_dir / "diagnostics.json").read_text())
    assert diag["num_forget_images"] == 24
    assert diag["num_classes"] == 10
    assert (diag_dir / "contact_sheet.png").is_file()

    # 2. build_retain_proxy.py
    ret_dir = work / "retain"
    _run_cli(
        "build_retain_proxy.py",
        [
            "--task-dir", str(task_dir),
            "--pool", str(pool_path),
            "--out-dir", str(ret_dir),
            "--per-class", "4",
            "--min-per-class", "3",
            "--dedup-cos", "0.99",
            "--audit-count", "8",
            "--mixup-fallback",
            "--batch-size", "16",
            "--device", "cpu",
        ],
        cwd=tmp_path,
    )
    retain_fit = ret_dir / "retain_fit.npy"
    retain_audit = ret_dir / "retain_audit.npy"
    assert retain_fit.is_file() and retain_audit.is_file()

    # 3. train_shadow_gold.py
    sg_path = work / "shadow_gold.pt"
    _run_cli(
        "train_shadow_gold.py",
        [
            "--task-dir", str(task_dir),
            "--retain", str(retain_fit),
            "--out", str(sg_path),
            "--num-models", "2",
            "--epochs", "1",
            "--bs", "8",
            "--device", "cpu",
            "--amp", "fp32",
        ],
        cwd=tmp_path,
    )
    assert sg_path.is_file() and sg_path.with_suffix(".json").is_file()

    # 4. build_oracle.py
    oracle_path = work / "oracle_targets.pt"
    _run_cli(
        "build_oracle.py",
        [
            "--task-dir", str(task_dir),
            "--out", str(oracle_path),
            "--shadow-gold", str(sg_path),
            "--num-weight-samples", "2",
            "--oof-splits", "3",
            "--batch-size", "8",
            "--device", "cpu",
        ],
        cwd=tmp_path,
    )
    assert oracle_path.is_file() and oracle_path.with_suffix(".json").is_file()

    # 5a. hybrid_unlearn.py (Full Hybrid + SSD warmstart + SalUn + Moment + NPO + Smooth + SCAR + HeadGaussian + EWC + Polish + Epoch saves)
    branch_full = work / "branch_full.pt"
    _run_cli(
        "hybrid_unlearn.py",
        [
            "--task-dir", str(task_dir),
            "--retain", str(retain_fit),
            "--retain-audit", str(retain_audit),
            "--oracle", str(oracle_path),
            "--shadow-gold", str(sg_path),
            "--out", str(branch_full),
            "--epochs", "1",
            "--bs", "8",
            "--eval-bs", "16",
            "--scope", "late_mlp",
            "--train-blocks", "2",
            "--ssd-warmstart",
            "--salun-ratio", "0.5",
            "--salun-mode", "relative_ssd",
            "--fisher-samples", "8",
            "--w-moment", "0.25",
            "--moment-mode", "wasserstein",
            "--w-npo", "0.15",
            "--w-smooth", "0.2",
            "--w-scar", "0.1",
            "--w-head-gaussian", "0.1",
            "--w-ewc", "0.5",
            "--rum-curriculum",
            "--polish-epochs", "1",
            "--save-every-epoch",
            "--device", "cpu",
            "--amp", "fp32",
        ],
        cwd=tmp_path,
    )
    _assert_valid_checkpoint(branch_full, model_path, task_dir)
    _assert_valid_checkpoint(work / "branch_full_epochs" / "epoch_01.pt", model_path, task_dir)
    full_metrics = json.loads(branch_full.with_suffix(".json").read_text())
    assert "final_score" in full_metrics

    # 5b. hybrid_unlearn.py (Mergeable LoRA + OGP + Prune warmstart)
    branch_lora = work / "branch_lora.pt"
    _run_cli(
        "hybrid_unlearn.py",
        [
            "--task-dir", str(task_dir),
            "--retain", str(retain_fit),
            "--retain-audit", str(retain_audit),
            "--oracle", str(oracle_path),
            "--out", str(branch_lora),
            "--epochs", "1",
            "--bs", "8",
            "--eval-bs", "16",
            "--train-blocks", "2",
            "--prune-warmstart",
            "--prune-sparsity", "0.10",
            "--lora-rank", "4",
            "--lora-alpha", "8.0",
            "--ogp-rank", "8",
            "--ogp-samples", "12",
            "--w-ewc", "0.0",
            "--polish-epochs", "1",
            "--device", "cpu",
            "--amp", "fp32",
        ],
        cwd=tmp_path,
    )
    _assert_valid_checkpoint(branch_lora, model_path, task_dir)

    # 6. kfold_unlearn.py
    kfold_dir = work / "kfold"
    _run_cli(
        "kfold_unlearn.py",
        [
            "--task-dir", str(task_dir),
            "--out-dir", str(kfold_dir),
            "--folds", "3",
            "--retain", str(retain_fit),
            "--retain-audit", str(retain_audit),
            "--oracle", str(oracle_path),
            "--shadow-gold", str(sg_path),
            "--merge-method", "ties",
            "--alphas", "0.8,1.0",
            "--epochs", "1",
            "--bs", "8",
            "--eval-bs", "16",
            "--train-blocks", "2",
            "--fisher-samples", "8",
            "--merged-polish-epochs", "1",
            "--device", "cpu",
            "--amp", "fp32",
        ],
        cwd=tmp_path,
    )
    kfold_ckpt = kfold_dir / "merged_best.pt"
    _assert_valid_checkpoint(kfold_ckpt, model_path, task_dir)
    assert (kfold_dir / "kfold_metrics.json").is_file()

    # 7. soup_and_eval.py
    soup_ckpt = work / "final_soup.pt"
    soup_json = work / "final_soup.json"
    _run_cli(
        "soup_and_eval.py",
        [
            "--task-dir", str(task_dir),
            "--models", str(branch_full), str(branch_lora), str(kfold_ckpt),
            "--search-weights",
            "--search-gammas",
            "--retain-audit", str(retain_audit),
            "--shadow-gold", str(sg_path),
            "--eval-batch-size", "16",
            "--out", str(soup_ckpt),
            "--metrics-out", str(soup_json),
            "--device", "cpu",
        ],
        cwd=tmp_path,
    )
    _assert_valid_checkpoint(soup_ckpt, model_path, task_dir)
    soup_meta = json.loads(soup_json.read_text())
    assert "final_score" in soup_meta and len(soup_meta["soup_weights"]) == 3


def test_slurm_scripts_shell_syntax() -> None:
    jobs_dir = UNLEARNING_DIR / "jobs"
    scripts = sorted(jobs_dir.glob("*.sh"))
    assert len(scripts) == 7
    for script in scripts:
        res = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True, check=False)
        assert res.returncode == 0, f"bash -n failed on {script}:\n{res.stderr}"
