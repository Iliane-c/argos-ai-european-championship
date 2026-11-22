"""Model registry, forensic stems, conditioning modules, and W1/W2/W3/W7 architectures."""
from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wl import GRID, MAX_BITS, ROOT_DIR, ensure_kit_on_path
from wl.data import count_input_channels

try:
    import segmentation_models_pytorch as smp
except ImportError:
    smp = None

try:
    import timm
except ImportError:
    timm = None


class ConvBlock(nn.Module):
    """Two 3x3 conv + GroupNorm + GELU blocks."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        groups = min(8, out_ch) if out_ch % min(8, out_ch) == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SRMConv2d(nn.Module):
    """Fixed 3-filter forensic high-pass residual bank applied to the first 3 RGB channels."""

    def __init__(self):
        super().__init__()
        kb = torch.tensor([[-1.0, 2.0, -1.0], [2.0, -4.0, 2.0], [-1.0, 2.0, -1.0]]) / 4.0
        lap = torch.tensor([[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]) / 4.0
        diag = torch.tensor([[-1.0, 0.0, -1.0], [0.0, 4.0, 0.0], [-1.0, 0.0, -1.0]]) / 4.0
        kernels = torch.stack([kb, lap, diag], dim=0).unsqueeze(1).repeat(1, 3, 1, 1) / 3.0
        self.register_buffer("weight", kernels)

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.conv2d(rgb, self.weight, padding=1) * 16.0, -1.0, 1.0)


class BayarConv2d(nn.Module):
    """Constrained forensic prediction-error convolution (center=-1, neighbors sum to +1)."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = kernel_size
        self.raw_weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.05)
        mask = torch.ones(1, 1, kernel_size, kernel_size)
        mask[:, :, kernel_size // 2, kernel_size // 2] = 0.0
        self.register_buffer("neighbor_mask", mask)

    def _constrained_weight(self) -> torch.Tensor:
        w = self.raw_weight * self.neighbor_mask
        denom = w.sum(dim=(-2, -1), keepdim=True).abs().clamp(min=1e-6)
        w = w / denom
        center = self.kernel_size // 2
        w_out = w.clone()
        w_out[:, :, center, center] = -1.0
        return w_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self._constrained_weight(), padding=self.kernel_size // 2)


class MessageConditioner(nn.Module):
    """Embed method ID and binary message into FiLM modulation parameters."""

    def __init__(self, feat_dim: int, mode: str = "message", hidden_dim: int = 128):
        super().__init__()
        self.mode = mode
        self.method_emb = nn.Embedding(3, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(MAX_BITS, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_film = nn.Linear(hidden_dim, feat_dim * 2)
        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_film.bias)

    def encode_context(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        cond = self.method_emb(batch["method_id"])
        if self.mode == "message" and "message_signed" in batch and "message_mask" in batch:
            msg = batch["message_signed"] * batch["message_mask"]
            cond = cond + self.msg_mlp(msg)
        return cond

    def forward(self, feat: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.mode == "none":
            return feat
        ctx = self.encode_context(batch)
        gamma, beta = self.to_film(ctx).chunk(2, dim=-1)
        return feat * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]


class MultiHeadOutput(nn.Module):
    """Standard 128x128 mask head plus optional kind, bit-map, and residual heads."""

    def __init__(self, feat_dim: int, heads: Sequence[str] = ("mask",), cond_mode: str = "none"):
        super().__init__()
        self.heads = set(heads)
        self.cond = MessageConditioner(feat_dim, mode=cond_mode) if cond_mode != "none" else None
        self.refine = ConvBlock(feat_dim, feat_dim)
        self.mask_head = nn.Conv2d(feat_dim, 1, kernel_size=1)
        self.kind_head = nn.Linear(feat_dim * 2, 1) if "kind" in self.heads else None
        self.bit_head = nn.Conv2d(feat_dim, MAX_BITS, kernel_size=1) if "bits" in self.heads else None
        self.residual_head = (
            nn.Sequential(
                nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
                nn.Conv2d(feat_dim, 3, kernel_size=3, padding=1),
            )
            if "residual" in self.heads
            else None
        )

    def forward(self, feat128: torch.Tensor, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if feat128.shape[-2:] != (GRID, GRID):
            feat128 = F.adaptive_avg_pool2d(feat128, (GRID, GRID))
        if self.cond is not None:
            feat128 = self.cond(feat128, batch)
        h = self.refine(feat128)
        out: dict[str, torch.Tensor] = {"logits": self.mask_head(h).squeeze(1)}
        if self.kind_head is not None:
            avg_p = h.mean(dim=(-2, -1))
            max_p = h.amax(dim=(-2, -1))
            out["kind_logits"] = self.kind_head(torch.cat([avg_p, max_p], dim=1)).squeeze(1)
        if self.bit_head is not None:
            out["bit_logits"] = self.bit_head(h)
        if self.residual_head is not None:
            out["residual"] = self.residual_head(h)
        return out


class TinyUNet(nn.Module):
    """Lightweight pure-PyTorch U-Net outputting 128x128 features without external downloads."""

    def __init__(
        self,
        in_ch: int,
        base_ch: int = 32,
        cond_mode: str = "none",
        heads: Sequence[str] = ("mask", "kind"),
    ):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2)       # 256
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2)   # 128
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, stride=2)   # 64
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, stride=2)  # 32

        self.up4 = nn.Conv2d(base_ch * 8 + base_ch * 8, base_ch * 4, kernel_size=1)
        self.dec4 = ConvBlock(base_ch * 4, base_ch * 4)
        self.up3 = nn.Conv2d(base_ch * 4 + base_ch * 4, base_ch * 2, kernel_size=1)
        self.dec3 = ConvBlock(base_ch * 2, base_ch * 2)
        self.head = MultiHeadOutput(base_ch * 2, heads=heads, cond_mode=cond_mode)

    def extract_features128(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)

        u4 = F.interpolate(b, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(self.up4(torch.cat([u4, e4], dim=1)))
        u3 = F.interpolate(d4, size=(GRID, GRID), mode="bilinear", align_corners=False)
        d3 = self.dec3(self.up3(torch.cat([u3, e3], dim=1)))
        return d3

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.head(self.extract_features128(batch["image"]), batch)


class SMPModelWrapper(nn.Module):
    """Wrapper around any segmentation_models_pytorch architecture and encoder."""

    def __init__(
        self,
        in_ch: int,
        arch: str = "Unet",
        encoder: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        feat_dim: int = 32,
        cond_mode: str = "none",
        heads: Sequence[str] = ("mask", "kind"),
    ):
        super().__init__()
        if smp is None:
            raise ImportError("segmentation_models_pytorch is required for model.name='smp'")
        self.backbone = smp.create_model(
            arch=arch,
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_ch,
            classes=feat_dim,
        )
        self.head = MultiHeadOutput(feat_dim, heads=heads, cond_mode=cond_mode)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feats = self.backbone(batch["image"])
        return self.head(feats, batch)


class TimmFPNSegmenter(nn.Module):
    """Multi-scale FPN decoder over any timm `features_only` backbone (HRNet, ConvNeXt, Swin, DINO)."""

    def __init__(
        self,
        in_ch: int,
        encoder: str = "hrnet_w48",
        pretrained: bool = True,
        fpn_dim: int = 128,
        cond_mode: str = "none",
        heads: Sequence[str] = ("mask", "kind"),
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for model.name='timm_fpn'")
        self.encoder = timm.create_model(
            encoder, pretrained=pretrained, features_only=True, in_chans=in_ch
        )
        chans = self.encoder.feature_info.channels()
        self.expected_chans = chans
        self.projs = nn.ModuleList([nn.Conv2d(c, fpn_dim, kernel_size=1) for c in chans])
        self.fuse = ConvBlock(fpn_dim, fpn_dim)
        self.head = MultiHeadOutput(fpn_dim, heads=heads, cond_mode=cond_mode)

    def extract_features128(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        fused = None
        for idx, (f, proj) in enumerate(zip(feats, self.projs)):
            if f.ndim == 4 and f.shape[1] != self.expected_chans[idx] and f.shape[-1] == self.expected_chans[idx]:
                f = f.permute(0, 3, 1, 2).contiguous()
            p = F.interpolate(proj(f), size=(GRID, GRID), mode="bilinear", align_corners=False)
            fused = p if fused is None else fused + p
        return self.fuse(fused)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.head(self.extract_features128(batch["image"]), batch)


class W1UnpooledDecoderNet(nn.Module):
    """W1: Unpooled watermark extractor producing spatial bit-agreement maps fused with SRM + FPN."""

    def __init__(
        self,
        in_ch: int,
        feat_dim: int = 64,
        use_kit_backbone: bool = False,
        heads: Sequence[str] = ("mask", "kind", "bits"),
    ):
        super().__init__()
        self.srm = SRMConv2d()
        self.use_kit_backbone = use_kit_backbone
        self.kit_detectors: dict[str, Any] = {}
        if use_kit_backbone:
            ensure_kit_on_path()
            from wmloc.watermarks import load_watermark

            models_dir = ROOT_DIR / "kit" / "models"
            self.ps_det = load_watermark("pixelseal", models_dir=models_dir, device="cpu").det
            self.tm_net = load_watermark("trustmark", models_dir=models_dir, device="cpu").net

        self.noise_stem = nn.Sequential(
            ConvBlock(in_ch + 3, feat_dim, stride=2),      # 256
            ConvBlock(feat_dim, feat_dim * 2, stride=2),   # 128
            ConvBlock(feat_dim * 2, feat_dim * 2),
        )
        self.bit_proj = nn.Conv2d(feat_dim * 2, MAX_BITS, kernel_size=1)
        self.agree_proj = nn.Conv2d(MAX_BITS, 16, kernel_size=1)
        self.decoder = ConvBlock(feat_dim * 2 + 16 + 2 + (1 if use_kit_backbone else 0), feat_dim)
        self.head = MultiHeadOutput(feat_dim, heads=heads, cond_mode="method")

    def _kit_agreement_map(self, rgb: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        bsz = rgb.shape[0]
        out = rgb.new_zeros((bsz, 1, GRID, GRID))
        if not self.use_kit_backbone:
            return out
        signed = batch["message_signed"]
        mask = batch["message_mask"]
        mid = batch["method_id"]

        ps_idx = (mid == 0).nonzero(as_tuple=True)[0]
        if ps_idx.numel() > 0:
            x_ps = F.interpolate(rgb[ps_idx], size=(256, 256), mode="bilinear", align_corners=False)
            x_ps = self.ps_det.preprocess(x_ps)
            emb = self.ps_det.pixel_decoder.output_upscaling(self.ps_det.convnext(x_ps))
            lin = self.ps_det.pixel_decoder.linear
            per_pos = torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None]
            logits = per_pos[:, 1:257]
            agree = (torch.tanh(logits * 0.5) * signed[ps_idx, :, None, None]).mean(dim=1, keepdim=True)
            out[ps_idx] = F.interpolate(agree, size=(GRID, GRID), mode="bilinear", align_corners=False).to(out.dtype)

        tm_idx = (mid == 1).nonzero(as_tuple=True)[0]
        if tm_idx.numel() > 0:
            x_tm = F.interpolate(rgb[tm_idx], size=(245, 245), mode="bilinear", align_corners=False) * 2.0 - 1.0
            n = self.tm_net
            h = n.maxpool(n.relu(n.bn1(n.conv1(x_tm))))
            for k in ("layer1", "layer2", "layer3", "layer4"):
                h = getattr(n, k)(h)
            logits = torch.einsum("bchw,oc->bohw", h, n.fc.weight) + n.fc.bias[None, :, None, None]
            s_tm = signed[tm_idx, :100, None, None]
            m_tm = mask[tm_idx, :100, None, None]
            agree = (torch.tanh(logits * 0.5) * s_tm * m_tm).sum(dim=1, keepdim=True) / m_tm.sum(dim=1, keepdim=True).clamp(min=1.0)
            out[tm_idx] = F.interpolate(agree, size=(GRID, GRID), mode="bilinear", align_corners=False).to(out.dtype)
        return out

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        srm = self.srm(batch["rgb"])
        x = torch.cat([batch["image"], srm], dim=1)
        vis128 = self.noise_stem(x)
        bit_logits = self.bit_proj(vis128)

        signed = batch["message_signed"][:, :, None, None]
        mask = batch["message_mask"][:, :, None, None]
        valid_cnt = mask.sum(dim=1, keepdim=True).clamp(min=1.0)

        prod = torch.tanh(bit_logits * 0.5) * signed * mask
        mean_agree = prod.sum(dim=1, keepdim=True) / valid_cnt
        cos_agree = F.cosine_similarity(torch.tanh(bit_logits * 0.5) * mask, signed * mask, dim=1).unsqueeze(1)
        learned_agree = self.agree_proj(prod)

        parts = [vis128, mean_agree, cos_agree, learned_agree]
        if self.use_kit_backbone:
            parts.append(self._kit_agreement_map(batch["rgb"], batch))

        fused = self.decoder(torch.cat(parts, dim=1))
        out = self.head(fused, batch)
        out["bit_logits"] = bit_logits
        return out


class W2WAMMultiTaskNet(nn.Module):
    """W2: Watermark Anything multi-task model predicting per-cell detection + message bits."""

    def __init__(
        self,
        in_ch: int,
        backbone_name: str = "tiny_unet",
        encoder: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        feat_dim: int = 64,
        heads: Sequence[str] = ("mask", "kind", "bits"),
    ):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "smp":
            if smp is None:
                raise ImportError("segmentation_models_pytorch required for w2_wam with backbone='smp'")
            self.backbone = smp.create_model(
                arch="Unet",
                encoder_name=encoder,
                encoder_weights=encoder_weights,
                in_channels=in_ch,
                classes=feat_dim,
            )
        elif backbone_name == "timm_fpn":
            self.backbone = TimmFPNSegmenter(in_ch=in_ch, encoder=encoder, pretrained=bool(encoder_weights), fpn_dim=feat_dim)
        else:
            self.backbone = TinyUNet(in_ch=in_ch, base_ch=max(32, feat_dim // 2), cond_mode="none", heads=("mask",))
            feat_dim = max(32, feat_dim // 2) * 2

        self.det_conv = nn.Conv2d(feat_dim, 1, kernel_size=3, padding=1)
        self.bit_conv = nn.Conv2d(feat_dim, MAX_BITS, kernel_size=1)
        self.kind_head = nn.Linear(feat_dim * 2, 1) if "kind" in heads else None
        self.alpha = nn.Parameter(torch.tensor(2.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def _get_feat128(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_name == "smp":
            return F.adaptive_avg_pool2d(self.backbone(x), (GRID, GRID))
        return self.backbone.extract_features128(x)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = self._get_feat128(batch["image"])
        det_logit = self.det_conv(feat).squeeze(1)
        bit_logits = self.bit_conv(feat)

        signed = batch["message_signed"][:, :, None, None]
        mask = batch["message_mask"][:, :, None, None]
        agree = (torch.tanh(bit_logits * 0.5) * signed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        out: dict[str, torch.Tensor] = {
            "logits": det_logit + self.alpha * agree + self.beta,
            "det_logits": det_logit,
            "bit_logits": bit_logits,
        }
        if self.kind_head is not None:
            pooled = torch.cat([feat.mean(dim=(-2, -1)), feat.amax(dim=(-2, -1))], dim=1)
            out["kind_logits"] = self.kind_head(pooled).squeeze(1)
        return out


class W3MessageConditionedNet(nn.Module):
    """W3: Referring-segmentation style segmenter using cross-attention to 16 message token groups."""

    def __init__(
        self,
        in_ch: int,
        backbone_name: str = "tiny_unet",
        arch: str = "Segformer",
        encoder: str = "mit_b2",
        encoder_weights: str | None = "imagenet",
        dim: int = 64,
        heads: Sequence[str] = ("mask", "kind"),
    ):
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "smp":
            if smp is None:
                raise ImportError("segmentation_models_pytorch required for w3_msg_cond with backbone='smp'")
            self.backbone = smp.create_model(
                arch=arch, encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_ch, classes=dim
            )
        else:
            self.backbone = TinyUNet(in_ch=in_ch, base_ch=max(32, dim // 2))
            dim = max(32, dim // 2) * 2

        self.dim = dim
        self.group_proj = nn.Linear(16, dim)
        self.method_emb = nn.Embedding(3, dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.head = MultiHeadOutput(dim, heads=heads, cond_mode="message")

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if self.backbone_name == "smp":
            feat = F.adaptive_avg_pool2d(self.backbone(batch["image"]), (GRID, GRID))
        else:
            feat = self.backbone.extract_features128(batch["image"])

        # Attend at 32x32 resolution for speed and upsample residual gating to 128x128.
        low = F.adaptive_avg_pool2d(feat, (32, 32))
        bsz, c, h, w = low.shape
        q = low.flatten(2).transpose(1, 2)  # [B, 1024, D]
        msg_groups = (batch["message_signed"] * batch["message_mask"]).view(bsz, 16, 16)
        kv = self.group_proj(msg_groups) + self.method_emb(batch["method_id"]).unsqueeze(1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        q_upd = self.norm(q + attn_out).transpose(1, 2).view(bsz, c, h, w)
        feat = feat + F.interpolate(q_upd, size=(GRID, GRID), mode="bilinear", align_corners=False)
        return self.head(feat, batch)


class W7DualStreamForensicNet(nn.Module):
    """W7: Dual-stream full-res noise branch (Bayar + SRM) fused with semantic segmentation stream."""

    def __init__(
        self,
        in_ch: int,
        sem_backbone: str = "tiny_unet",
        arch: str = "Segformer",
        encoder: str = "mit_b2",
        encoder_weights: str | None = "imagenet",
        feat_dim: int = 64,
        cond_mode: str = "message",
        heads: Sequence[str] = ("mask", "kind"),
    ):
        super().__init__()
        self.bayar = BayarConv2d(3, 3, kernel_size=5)
        self.srm = SRMConv2d()
        # Noise branch avoids aggressive early pooling to preserve pixel-grid watermark signals.
        self.noise_stream = nn.Sequential(
            ConvBlock(6 + max(0, in_ch - 3), 32, stride=1),
            ConvBlock(32, 48, stride=2),
            ConvBlock(48, feat_dim, stride=2),
        )

        self.sem_backbone_name = sem_backbone
        if sem_backbone == "smp":
            if smp is None:
                raise ImportError("segmentation_models_pytorch required for w7 with sem_backbone='smp'")
            self.sem_stream = smp.create_model(
                arch=arch, encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_ch, classes=feat_dim
            )
        elif sem_backbone == "timm_fpn":
            self.sem_stream = TimmFPNSegmenter(in_ch=in_ch, encoder=encoder, pretrained=bool(encoder_weights), fpn_dim=feat_dim)
        else:
            self.sem_stream = TinyUNet(in_ch=in_ch, base_ch=max(32, feat_dim // 2))
            feat_dim = max(32, feat_dim // 2) * 2

        self.noise_proj = nn.Conv2d(self.noise_stream[-1].net[-3].out_channels, feat_dim, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv2d(feat_dim, feat_dim, kernel_size=1), nn.Sigmoid())
        self.fuse = ConvBlock(feat_dim * 3, feat_dim)
        self.head = MultiHeadOutput(feat_dim, heads=heads, cond_mode=cond_mode)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        rgb = batch["rgb"]
        extra = batch["image"][:, 3:]
        noise_in = torch.cat([self.bayar(rgb), self.srm(rgb), extra], dim=1)
        noise_f = F.adaptive_avg_pool2d(self.noise_proj(self.noise_stream(noise_in)), (GRID, GRID))

        if self.sem_backbone_name == "smp":
            sem_f = F.adaptive_avg_pool2d(self.sem_stream(batch["image"]), (GRID, GRID))
        else:
            sem_f = self.sem_stream.extract_features128(batch["image"])

        gated_sem = sem_f * self.gate(noise_f)
        fused = self.fuse(torch.cat([noise_f, sem_f, gated_sem], dim=1))
        return self.head(fused, batch)


_KIT_CACHE: dict[str, dict[str, Any]] = {}  # module-level so EMA deepcopies do not duplicate the extractors


def _kit_models(device: torch.device, methods: Sequence[str]) -> dict[str, Any]:
    key = str(device)
    if key not in _KIT_CACHE:
        from wl.features import load_kit_models

        _KIT_CACHE[key] = load_kit_models(list(methods), device=key)
    return _KIT_CACHE[key]


class W8DecoderInLoopNet(W2WAMMultiTaskNet):
    """W8: W2 + the official extractors run inside the forward pass (frozen).

    E141 fed the decoder evidence as cached maps: constant on synthetic shards and misaligned with the
    shift augmentation, so the network learned to ignore it. Here the PixelSeal / TrustMark extractors
    run on the (augmented) input itself, so the evidence is always present and pixel-aligned. Their
    per-position bit logits are compared with the known message (WAM-style agreement, sub-stride
    interleaved to a finer grid) and fused with the learned features; a residual path adds the raw
    agreement to the final logit so training starts from the decoder's own localization.
    """

    def __init__(self, *args: Any, kit_methods: Sequence[str] = ("pixelseal", "trustmark"), pp_shifts: int = 2,
                 ev_dim: int = 16, **kwargs: Any):
        super().__init__(*args, **kwargs)
        feat_dim = self.det_conv.in_channels
        self.kit_methods = tuple(kit_methods)
        self.pp_shifts = int(pp_shifts)
        self.ev_proj = nn.Conv2d(MAX_BITS, ev_dim, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_dim + ev_dim + 2, feat_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        if self.kind_head is not None:
            self.kind_head = nn.Linear(feat_dim * 2, 1)
        self.gamma = nn.Parameter(torch.tensor(4.0))

    @staticmethod
    def _feat(wm: Any, x: torch.Tensor, nbits: int) -> torch.Tensor:
        out = wm.features(x)["per_position_logits"]
        if out.dim() == 3:  # unbatched output
            out = out[None]
        if out.shape[0] != x.shape[0]:  # extractor wrapper only handles one image at a time
            out = torch.cat([wm.features(x[i : i + 1])["per_position_logits"].reshape(1, *out.shape[-3:]) for i in range(x.shape[0])])
        return out.float()[:, :nbits].to(x.device)

    @torch.no_grad()
    def _per_position(self, wm: Any, x: torch.Tensor, nbits: int) -> torch.Tensor:
        """Per-position bit logits at PP_SHIFTS x finer stride (shift-and-interleave, as in wl.acontrario)."""
        size = x.shape[-1]
        pp0 = self._feat(wm, x, nbits)
        hf, wf = pp0.shape[-2:]
        s = self.pp_shifts
        if s <= 1:
            return pp0
        sy_step, sx_step = size // hf, size // wf
        hr = pp0.new_zeros((x.shape[0], nbits, hf * s, wf * s))
        ar_h = torch.arange(hf, device=x.device)
        ar_w = torch.arange(wf, device=x.device)
        for sy in range(s):
            for sx in range(s):
                dy, dx = sy * sy_step // s, sx * sx_step // s
                pp = pp0 if (dy, dx) == (0, 0) else self._feat(wm, torch.roll(x, shifts=(dy, dx), dims=(-2, -1)), nbits)
                rows = ((ar_h * sy_step + sy_step // 2 - dy) % size) // (sy_step // s)
                cols = ((ar_w * sx_step + sx_step // 2 - dx) % size) // (sx_step // s)
                hr[:, :, rows[:, None], cols[None, :]] = pp
        return hr

    def _evidence(self, batch: dict[str, torch.Tensor], like: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from wl import METHOD_TO_ID, NBITS

        rgb = batch["rgb"]
        bsz = rgb.shape[0]
        ev = like.new_zeros((bsz, self.ev_proj.out_channels + 2, GRID, GRID))
        mid = batch["method_id"]
        kit = _kit_models(rgb.device, self.kit_methods)
        for m in self.kit_methods:
            idx = (mid == METHOD_TO_ID[m]).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            nb = NBITS[m]
            with torch.autocast(device_type=rgb.device.type, enabled=False):
                logits = self._per_position(kit[m], rgb[idx].float(), nb)
            t = torch.tanh(logits * 0.5)
            prod = t * batch["message_signed"][idx, :nb, None, None].float()
            agree = prod.mean(dim=1, keepdim=True)
            strength = t.abs().mean(dim=1, keepdim=True)
            pad = prod.new_zeros((prod.shape[0], MAX_BITS) + prod.shape[-2:])
            pad[:, :nb] = prod
            parts = torch.cat([agree * 4.0, strength, self.ev_proj(pad)], dim=1)
            ev[idx] = F.interpolate(parts, size=(GRID, GRID), mode="bilinear", align_corners=False).to(ev.dtype)
        return ev, ev[:, 0] / 4.0

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = self._get_feat128(batch["image"])
        ev, agree_kit = self._evidence(batch, feat)
        fused = self.fuse(torch.cat([feat, ev], dim=1))
        det_logit = self.det_conv(fused).squeeze(1)
        bit_logits = self.bit_conv(fused)

        signed = batch["message_signed"][:, :, None, None]
        mask = batch["message_mask"][:, :, None, None]
        agree = (torch.tanh(bit_logits * 0.5) * signed * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        out: dict[str, torch.Tensor] = {
            "logits": det_logit + self.alpha * agree + self.gamma * agree_kit + self.beta,
            "det_logits": det_logit,
            "bit_logits": bit_logits,
        }
        if self.kind_head is not None:
            pooled = torch.cat([fused.mean(dim=(-2, -1)), fused.amax(dim=(-2, -1))], dim=1)
            out["kind_logits"] = self.kind_head(pooled).squeeze(1)
        return out


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Instantiate model from experiment configuration dictionary."""
    mcfg = cfg.get("model", {})
    channels = cfg.get("input", {}).get("channels", ["rgb"])
    in_ch = count_input_channels(channels)
    name = str(mcfg.get("name", "tiny_unet"))
    cond = str(mcfg.get("cond", "none"))
    heads = tuple(mcfg.get("heads", ["mask", "kind"]))

    if name == "tiny_unet":
        return TinyUNet(
            in_ch=in_ch,
            base_ch=int(mcfg.get("base_ch", 32)),
            cond_mode=cond,
            heads=heads,
        )
    if name == "smp":
        return SMPModelWrapper(
            in_ch=in_ch,
            arch=str(mcfg.get("arch", "Unet")),
            encoder=str(mcfg.get("encoder", "resnet34")),
            encoder_weights=mcfg.get("encoder_weights", "imagenet"),
            feat_dim=int(mcfg.get("feat_dim", 32)),
            cond_mode=cond,
            heads=heads,
        )
    if name == "timm_fpn":
        return TimmFPNSegmenter(
            in_ch=in_ch,
            encoder=str(mcfg.get("encoder", "hrnet_w48")),
            pretrained=bool(mcfg.get("pretrained", True)),
            fpn_dim=int(mcfg.get("feat_dim", 128)),
            cond_mode=cond,
            heads=heads,
        )
    if name == "w1_unpooled":
        return W1UnpooledDecoderNet(
            in_ch=in_ch,
            feat_dim=int(mcfg.get("feat_dim", 64)),
            use_kit_backbone=bool(mcfg.get("use_kit_backbone", False)),
            heads=heads,
        )
    if name == "w2_wam":
        return W2WAMMultiTaskNet(
            in_ch=in_ch,
            backbone_name=str(mcfg.get("backbone", "tiny_unet")),
            encoder=str(mcfg.get("encoder", "resnet34")),
            encoder_weights=mcfg.get("encoder_weights", "imagenet"),
            feat_dim=int(mcfg.get("feat_dim", 64)),
            heads=heads,
        )
    if name == "w8_decoder_loop":
        return W8DecoderInLoopNet(
            in_ch=in_ch,
            backbone_name=str(mcfg.get("backbone", "smp")),
            encoder=str(mcfg.get("encoder", "resnet50")),
            encoder_weights=mcfg.get("encoder_weights", "imagenet"),
            feat_dim=int(mcfg.get("feat_dim", 64)),
            heads=heads,
            kit_methods=tuple(mcfg.get("kit_methods", ["pixelseal", "trustmark"])),
            pp_shifts=int(mcfg.get("pp_shifts", 2)),
            ev_dim=int(mcfg.get("ev_dim", 16)),
        )
    if name == "w3_msg_cond":
        return W3MessageConditionedNet(
            in_ch=in_ch,
            backbone_name=str(mcfg.get("backbone", "tiny_unet")),
            arch=str(mcfg.get("arch", "Segformer")),
            encoder=str(mcfg.get("encoder", "mit_b2")),
            encoder_weights=mcfg.get("encoder_weights", "imagenet"),
            dim=int(mcfg.get("feat_dim", 64)),
            heads=heads,
        )
    if name == "w7_dual_stream":
        return W7DualStreamForensicNet(
            in_ch=in_ch,
            sem_backbone=str(mcfg.get("sem_backbone", "tiny_unet")),
            arch=str(mcfg.get("arch", "Segformer")),
            encoder=str(mcfg.get("encoder", "mit_b2")),
            encoder_weights=mcfg.get("encoder_weights", "imagenet"),
            feat_dim=int(mcfg.get("feat_dim", 64)),
            cond_mode=cond,
            heads=heads,
        )
    raise ValueError(f"Unknown model name: {name!r}")
