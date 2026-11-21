"""T3-010: theta(g) = theta_orig + g * (theta_unlearned - theta_orig); report stats per g, optionally export.

Run from the directory holding task_template.py, model.pt and forget_images.npz.
  python interpolate.py UNLEARNED.pt --gammas 1.25,1.5 [--retain coco224.npy] [--export DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import task_template as tt

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("unlearned", type=Path)
p.add_argument("--gammas", required=True)
p.add_argument("--retain", type=Path, help="uint8 .npy proxy images; the last 500 are used for drift")
p.add_argument("--export", type=Path)
args = p.parse_args()
dev = "cuda"

imgs = tt.load_forget_images()
sub = torch.arange(0, len(imgs), 5)
audit = torch.from_numpy(np.load(args.retain)[-500:]) if args.retain else None
sd0 = torch.load(tt.MODEL_FILE, map_location="cpu", weights_only=True)
sdu = torch.load(args.unlearned, map_location="cpu", weights_only=True)
model = tt.build_model().to(dev).eval()


def blur5(x):
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), torch.ones(3, 1, 5, 5, device=dev) / 25, groups=3)


@torch.no_grad()
def logits(u8, view=lambda x: x):
    out = []
    for i in range(0, len(u8), 128):
        x = u8[i:i + 128].to(dev).permute(0, 3, 1, 2).float() / 255
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model((view(x) - 0.5) / 0.5).float().cpu())
    return torch.cat(out)


def kl2(za, zb):
    pa, pb = F.softmax(za / 2, 1), F.softmax(zb / 2, 1)
    return (pa * (pa.clamp_min(1e-30).log() - pb.clamp_min(1e-30).log())).sum(1).mean().item()


model.load_state_dict(sd0, strict=True)
z0, zb0 = logits(imgs), logits(imgs[sub], blur5)
za0 = logits(audit) if audit is not None else None
y = z0.argmax(1)
for g in [float(s) for s in args.gammas.split(",")]:
    sd = {k: (sd0[k] + g * (sdu[k] - sd0[k])).float().contiguous() for k in sd0}
    model.load_state_dict(sd, strict=True)
    zf, zb = logits(imgs), logits(imgs[sub], blur5)
    ce, ceb = F.cross_entropy(zf, y, reduction="none"), F.cross_entropy(zb, y[sub], reduction="none")
    line = (f"g={g:4.2f} clean CE {ce.mean():.3f}/{ce.std():.3f} acc {(zf.argmax(1) == y).float().mean():.3f} "
            f"KL {kl2(z0, zf):.3f} | blur CE {ceb.mean():.3f}/{ceb.std():.3f} "
            f"acc {(zb.argmax(1) == y[sub]).float().mean():.3f} KL {kl2(zb0, zb):.3f}")
    if za0 is not None:
        za = logits(audit)
        line += f" | retain KL {kl2(za0, za):.4f} agree {(za.argmax(1) == za0.argmax(1)).float().mean():.3f}"
    print(line, flush=True)
    if args.export:
        assert all(torch.isfinite(v).all() for v in sd.values())
        out = args.export / f"{args.unlearned.stem}_g{g:g}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, out)
        tt.build_model().load_state_dict(torch.load(out, weights_only=True), strict=True)
        assert out.stat().st_size < 450_000_000
        print("  exported", out, out.stat().st_size, flush=True)
