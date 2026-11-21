"""Shared TinyViT mock and synthetic Task 3 environment fixtures for CPU tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

UNLEARNING_DIR = Path(__file__).resolve().parents[1]
if str(UNLEARNING_DIR) not in sys.path:
    sys.path.insert(0, str(UNLEARNING_DIR))

os.environ.setdefault("TMPDIR", "/tmp")


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 8, embed_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class TinyAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TinyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, init_values: float = 1e-4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim)
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class TinyViT(nn.Module):
    """Lightweight ViT mirroring timm vit_base_patch16_224 (LayerScale) module & tensor names."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 8,
        embed_dim: int = 32,
        depth: int = 4,
        num_heads: int = 4,
        num_classes: int = 10,
        init_values: float = 1e-4,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TinyBlock(embed_dim, num_heads=num_heads, init_values=init_values) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._grad_checkpointing = False
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self._grad_checkpointing = bool(enable)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls_tok = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tok, x), dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        feat = x[:, 0]
        return feat if pre_logits else self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x), pre_logits=False)


_MOCK_TASK_TEMPLATE = '''"""Mock task_template.py for local CPU testing."""
import torch
import torch.nn as nn

NUM_CLASSES = 10
IMAGE_SIZE = 32
EXPECTED_FORGET_IMAGES = 24


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 8, embed_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-4) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), float(init_values)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class TinyAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class TinyMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TinyBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, init_values: float = 1e-4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyAttention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = TinyMLP(dim)
        self.ls2 = LayerScale(dim, init_values=init_values)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class TinyViT(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 8, embed_dim: int = 32, depth: int = 4, num_heads: int = 4, num_classes: int = 10, init_values: float = 1e-4) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([TinyBlock(embed_dim, num_heads=num_heads, init_values=init_values) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self._grad_checkpointing = False

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self._grad_checkpointing = bool(enable)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        feat = x[:, 0]
        return feat if pre_logits else self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x), pre_logits=False)


def build_model() -> nn.Module:
    return TinyViT(img_size=IMAGE_SIZE, patch_size=8, embed_dim=32, depth=4, num_heads=4, num_classes=NUM_CLASSES, init_values=1e-4)
'''


def _make_class_images(labels: np.ndarray, rng: np.random.Generator, size: int = 32, noise_scale: float = 28.0) -> np.ndarray:
    imgs = rng.normal(loc=128.0, scale=noise_scale, size=(len(labels), size, size, 3))
    for i, c in enumerate(labels):
        # Distinct spatial block + color signature per class
        y0 = int((c % 5) * (size // 5))
        x0 = int((c // 5) * (size // 2))
        imgs[i, y0 : y0 + 6, x0 : x0 + 10, c % 3] += 95.0
        imgs[i, :, :, (c + 1) % 3] -= 25.0
    return np.clip(np.round(imgs), 0, 255).astype(np.uint8)


@pytest.fixture()
def tiny_model() -> TinyViT:
    torch.manual_seed(7)
    return TinyViT(img_size=32, patch_size=8, embed_dim=32, depth=4, num_heads=4, num_classes=10, init_values=1e-4).eval()


@pytest.fixture()
def synthetic_task_env(tmp_path: Path) -> dict[str, Path]:
    task_dir = tmp_path / "task3_mock"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task_template.py").write_text(_MOCK_TASK_TEMPLATE)

    rng = np.random.default_rng(42)
    forget_labels = np.arange(24) % 10
    forget_np = _make_class_images(forget_labels, rng, size=32, noise_scale=22.0)

    retain_labels = np.arange(40) % 10
    retain_np = _make_class_images(retain_labels, rng, size=32, noise_scale=42.0)

    torch.manual_seed(123)
    model = TinyViT(img_size=32, patch_size=8, embed_dim=32, depth=4, num_heads=4, num_classes=10, init_values=1e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-2)
    x_tr = (torch.from_numpy(forget_np).permute(0, 3, 1, 2).float() / 255.0 - 0.5) / 0.5
    y_tr = torch.from_numpy(forget_labels).long()
    model.train()
    for _ in range(25):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x_tr), y_tr)
        loss.backward()
        opt.step()
    model.eval()

    model_path = task_dir / "model.pt"
    state = {k: v.detach().cpu().float().contiguous() for k, v in model.state_dict().items()}
    torch.save(state, model_path)

    forget_path = task_dir / "forget_images.npz"
    np.savez(forget_path, images=forget_np)

    pool_path = task_dir / "retain_pool.npy"
    np.save(pool_path, retain_np)

    return {
        "task_dir": task_dir,
        "model_path": model_path,
        "forget_path": forget_path,
        "pool_path": pool_path,
    }
