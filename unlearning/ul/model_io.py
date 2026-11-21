"""Model construction, image loading, batched inference, and submission validation."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import torch.nn as nn

NUM_CLASSES = 100
IMAGE_SIZE = 224
EXPECTED_FORGET_IMAGES = 3500
EVAL_TEMPERATURE = 2.0
MAX_SUBMISSION_BYTES = 450_000_000


def resolve_device(device: str | None = None) -> str:
    """Resolve 'auto'/None to 'cuda' if available else 'cpu'."""
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" or device == "cuda" or device.startswith("cuda:"):
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}' but CUDA is unavailable.")
        return device
    raise ValueError(f"Unsupported device specification: '{device}'")


def resolve_amp_dtype(amp: str, device: str) -> torch.dtype | None:
    """Resolve AMP precision dtype for CUDA; CPU/fp32 runs in full float32."""
    if not str(device).startswith("cuda") or amp == "fp32":
        return None
    if amp == "bf16":
        return torch.bfloat16
    if amp == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported amp mode '{amp}'; expected 'bf16', 'fp16', or 'fp32'.")


def load_task_template(task_dir: Path | None = None) -> ModuleType | None:
    """Import task_template from task_dir or sys.path if present."""
    if task_dir is not None:
        resolved = Path(task_dir).resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Task directory does not exist: {resolved}")
        resolved_str = str(resolved)
        if resolved_str not in sys.path:
            sys.path.insert(0, resolved_str)
        existing = sys.modules.get("task_template")
        if existing is not None:
            existing_file = getattr(existing, "__file__", None)
            if existing_file is not None and Path(existing_file).resolve().parent != resolved:
                del sys.modules["task_template"]
    try:
        return importlib.import_module("task_template")
    except ModuleNotFoundError as exc:
        if exc.name == "task_template":
            return None
        raise


def build_model(
    num_classes: int | None = None,
    init_values: float = 1e-4,
    task_dir: Path | None = None,
) -> nn.Module:
    """Build ViT-B/16 matching task_template.py (or custom num_classes via timm)."""
    tt = load_task_template(task_dir)
    if tt is not None:
        tt_classes = getattr(tt, "NUM_CLASSES", NUM_CLASSES)
        if num_classes is None or num_classes == tt_classes:
            return tt.build_model()
    import timm

    resolved_classes = NUM_CLASSES if num_classes is None else num_classes
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=resolved_classes,
        init_values=init_values,
    )


def load_model(
    path: Path,
    device: str = "cpu",
    model_factory: Callable[[], nn.Module] | None = None,
    task_dir: Path | None = None,
) -> nn.Module:
    """Load a weights-only state_dict strictly into the model architecture."""
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    model = model_factory() if model_factory is not None else build_model(task_dir=task_dir)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


def load_images_uint8(path: Path, expected_count: int | None = None) -> torch.Tensor:
    """Load uint8 [N, H, W, 3] images from .npz (key 'images') or .npy."""
    img_path = Path(path)
    if not img_path.is_file():
        raise FileNotFoundError(f"Image archive not found: {img_path}")
    if img_path.suffix == ".npz":
        with np.load(img_path, allow_pickle=False) as data:
            if "images" not in data.files:
                raise KeyError(f"{img_path} missing 'images' key; found {data.files}")
            arr = data["images"].copy()
    elif img_path.suffix == ".npy":
        arr = np.load(img_path, allow_pickle=False)
        if arr.dtype != np.uint8:
            raise ValueError(f"Expected uint8 pixels in {img_path}, got {arr.dtype}")
        arr = arr.copy()
    else:
        raise ValueError(f"Unsupported image file extension '{img_path.suffix}' for {img_path}")

    images = torch.from_numpy(arr)
    if images.dtype != torch.uint8:
        raise ValueError(f"Expected uint8 tensor, got {images.dtype}")
    if images.ndim != 4:
        raise ValueError(f"Expected 4D [N, H, W, 3] tensor, got shape {tuple(images.shape)}")
    if images.shape[-1] != 3 and images.shape[1] == 3:
        images = images.permute(0, 2, 3, 1).contiguous()
    if images.shape[-1] != 3 or images.shape[1] != images.shape[2]:
        raise ValueError(f"Expected square RGB [N, H, H, 3] images, got {tuple(images.shape)}")
    if expected_count is not None and len(images) != expected_count:
        raise ValueError(f"Expected {expected_count} images in {img_path}, got {len(images)}")
    return images


def to_float01(images_uint8: torch.Tensor, device: str) -> torch.Tensor:
    """Convert [B, H, W, 3] uint8 NHWC tensor to [B, 3, H, W] float32 in [0, 1]."""
    if images_uint8.dtype != torch.uint8:
        raise ValueError(f"to_float01 expects uint8 tensor, got {images_uint8.dtype}")
    return images_uint8.to(device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0


def normalize_images(images_01: torch.Tensor) -> torch.Tensor:
    """Apply server-exact normalization: [0, 1] -> [-1, 1]."""
    return (images_01 - 0.5) / 0.5


def preprocess(images_uint8: torch.Tensor, device: str) -> torch.Tensor:
    """Convert uint8 [B, H, W, 3] images into normalized [-1, 1] NCHW tensors."""
    return normalize_images(to_float01(images_uint8, device))


def forward_features_and_logits(
    model: nn.Module, x_norm: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract pre-logit pooled ViT features and classification logits."""
    if not (hasattr(model, "forward_features") and hasattr(model, "forward_head") and hasattr(model, "head")):
        raise AttributeError(
            "Model must expose ViT methods forward_features, forward_head, and head."
        )
    features = model.forward_head(model.forward_features(x_norm), pre_logits=True)
    logits = model.head(features)
    return features, logits


def _autocast_ctx(device: str, use_fp16_on_cuda: bool):
    enabled = use_fp16_on_cuda and str(device).startswith("cuda")
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    images_uint8: torch.Tensor,
    device: str,
    batch_size: int = 128,
    view_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    use_fp16_on_cuda: bool = True,
) -> torch.Tensor:
    """Compute float32 CPU logits [N, C] matching server eval + CUDA fp16 autocast."""
    was_training = model.training
    model.eval()
    outputs: list[torch.Tensor] = []
    try:
        for batch in images_uint8.split(batch_size):
            x01 = to_float01(batch, device)
            if view_fn is not None:
                x01 = view_fn(x01)
            x_norm = normalize_images(x01)
            with _autocast_ctx(device, use_fp16_on_cuda):
                outputs.append(model(x_norm).float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def extract_features_and_logits(
    model: nn.Module,
    images_uint8: torch.Tensor,
    device: str,
    batch_size: int = 128,
    view_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    use_fp16_on_cuda: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute float32 CPU (pre-logit features [N, D], logits [N, C])."""
    was_training = model.training
    model.eval()
    feats_out: list[torch.Tensor] = []
    logits_out: list[torch.Tensor] = []
    try:
        for batch in images_uint8.split(batch_size):
            x01 = to_float01(batch, device)
            if view_fn is not None:
                x01 = view_fn(x01)
            x_norm = normalize_images(x01)
            with _autocast_ctx(device, use_fp16_on_cuda):
                feat, logits = forward_features_and_logits(model, x_norm)
            feats_out.append(feat.float().cpu())
            logits_out.append(logits.float().cpu())
    finally:
        model.train(was_training)
    return torch.cat(feats_out, dim=0), torch.cat(logits_out, dim=0)


def validate_and_save_submission(
    model_or_state: nn.Module | Mapping[str, torch.Tensor],
    out_path: Path,
    reference_state: Mapping[str, torch.Tensor] | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    max_bytes: int = MAX_SUBMISSION_BYTES,
    verify_forward_shape: tuple[int, int, int, int] | None = None,
) -> int:
    """Validate float32/finiteness/schema, atomically save state_dict, and verify strict reload."""
    raw_state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    clean_state: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        t = tensor.detach().cpu().float().contiguous()
        if t.dtype != torch.float32:
            raise ValueError(f"Parameter '{name}' must be float32, got {t.dtype}")
        if not torch.isfinite(t).all():
            raise ValueError(f"Parameter '{name}' contains NaN or Inf values")
        clean_state[name] = t

    if reference_state is not None:
        ref_keys = set(reference_state.keys())
        sub_keys = set(clean_state.keys())
        if ref_keys != sub_keys:
            missing = sorted(ref_keys - sub_keys)
            extra = sorted(sub_keys - ref_keys)
            raise ValueError(f"State dict key mismatch: missing={missing[:5]}, extra={extra[:5]}")
        for name, ref_tensor in reference_state.items():
            if clean_state[name].shape != ref_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': got {tuple(clean_state[name].shape)}, "
                    f"expected {tuple(ref_tensor.shape)}"
                )

    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f".{dest.name}.tmp")
    try:
        torch.save(clean_state, tmp_path)
        size_bytes = tmp_path.stat().st_size
        if size_bytes > max_bytes:
            raise ValueError(
                f"Checkpoint size {size_bytes:,} bytes exceeds limit {max_bytes:,} bytes"
            )
        tmp_path.replace(dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    if model_factory is not None:
        reloaded = model_factory()
        reloaded.load_state_dict(torch.load(dest, map_location="cpu", weights_only=True), strict=True)
        reloaded.eval()
        if verify_forward_shape is not None:
            with torch.no_grad():
                probe_out = reloaded(torch.zeros(verify_forward_shape, dtype=torch.float32))
            if not torch.isfinite(probe_out).all():
                raise ValueError("Reloaded checkpoint produced non-finite dummy forward outputs")

    return size_bytes


def add_common_io_args(parser) -> None:
    """Register standard --task-dir, --model, --forget, --device, --amp CLI arguments."""
    parser.add_argument("--task-dir", type=Path, default=None, help="Directory holding task_template.py")
    parser.add_argument("--model", type=Path, default=None, help="Path to original model.pt")
    parser.add_argument("--forget", type=Path, default=None, help="Path to forget_images.npz")
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, or cuda[:idx]")
    parser.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16")


def resolve_task_paths(args) -> tuple[Path | None, Path, Path]:
    """Resolve (task_dir, model_path, forget_path) using task_dir defaults when omitted."""
    task_dir = Path(args.task_dir) if args.task_dir is not None else None
    base = task_dir if task_dir is not None else Path(".")
    model_path = Path(args.model) if args.model is not None else base / "model.pt"
    forget_path = Path(args.forget) if args.forget is not None else base / "forget_images.npz"
    return task_dir, model_path, forget_path


def infer_vit_dims(model: nn.Module, images_uint8: torch.Tensor | None = None) -> tuple[int, int, int]:
    """Infer (num_classes, img_size, patch_size) from a ViT model and optional images."""
    num_classes = int(model.head.out_features) if hasattr(model, "head") else NUM_CLASSES
    img_size = int(images_uint8.shape[1]) if images_uint8 is not None else IMAGE_SIZE
    patch_size = 16
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
        ks = model.patch_embed.proj.kernel_size
        patch_size = int(ks[0] if isinstance(ks, tuple) else ks)
    return num_classes, img_size, patch_size
