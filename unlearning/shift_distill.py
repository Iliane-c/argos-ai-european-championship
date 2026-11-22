"""T3-010: distil the original model into "forgotten" targets on the forget set (delta-shift).

For every augmented view v of forget image i the target logits are
    t(v) = teacher(v) - delta_i(v) * e_y        (y = teacher argmax on the clean image)
i.e. a "memorisation bonus" is removed from the true class only, so wrong-class mass
goes to the teacher's own runner-ups. Retention: KD from the teacher at T=2 on proxy
images (uint8 array [N, 224, 224, 3]). Run from the directory holding task_template.py.

Options for the follow-up experiments:
  --view-ratio      delta_i(v) = delta_i * clip(margin_v / margin_clean, 0, 1)
                    (memorisation transfers to an altered view in proportion to its margin)
  --margin-power p  delta_i = delta * (margin_i / median margin) ** p
  --soften eps      mix eps of a per-class confusion prior into the targets (gold is an ensemble)
  --train-blocks k  train only the last k ViT blocks + final norm + head
  --target-scale s  multiply the shifted target logits by s (>1: confidently right or confidently wrong,
                    like a model on images it never saw)
  --split           train on 80% of the forget set, report the held-out 20% (leakage check)
"""
import argparse
import io
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import task_template as tt

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--retain", type=Path, required=True, help="uint8 .npy [N,224,224,3] proxy retain images")
p.add_argument("--retain-audit", type=int, default=500, help="last N retain images are held out for drift")
p.add_argument("--delta", type=float, default=12.0)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--epochs", type=int, default=8)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--retain-weight", type=float, default=1.0)
p.add_argument("--view-ratio", action="store_true")
p.add_argument("--margin-power", type=float, default=0.0)
p.add_argument("--soften", type=float, default=0.0)
p.add_argument("--train-blocks", type=int, default=12)
p.add_argument("--target-scale", type=float, default=1.0)
p.add_argument("--split", action="store_true")
p.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16", help="fp16 on Turing (no bf16)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

dev = "cuda"
torch.manual_seed(args.seed)
amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[args.amp]
imgs = tt.load_forget_images()
retain = torch.from_numpy(np.load(args.retain))
r_fit, r_aud = retain[:-args.retain_audit], retain[-args.retain_audit:]

teacher = tt.load_model().eval()
for q in teacher.parameters():
    q.requires_grad_(False)
student = tt.load_model()
student.set_grad_checkpointing(True)
for q in student.parameters():
    q.requires_grad_(False)
for m in [*student.blocks[12 - args.train_blocks:], student.norm, student.head]:
    for q in m.parameters():
        q.requires_grad_(True)


def to01(u8):
    return u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0


def norm(x):
    return (x - 0.5) / 0.5


@torch.no_grad()
def logits(model, u8, view=lambda x: x):
    out = []
    for i in range(0, len(u8), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(norm(view(to01(u8[i:i + 128])))).float().cpu())
    return torch.cat(out)


def true_margin(z, y):
    zy = z.gather(1, y[:, None]).squeeze(1)
    return zy - z.scatter(1, y[:, None], -1e9).max(1).values


z0 = logits(teacher, imgs)
y_all = z0.argmax(1)
m_clean = true_margin(z0, y_all).clamp_min(1e-3)
delta_i = args.delta * (m_clean / m_clean.median()) ** args.margin_power
# per-class confusion prior: mean T=1 alternative distribution of the teacher, true class masked
alt = F.softmax(z0.scatter(1, y_all[:, None], -1e9), 1)
prior = torch.zeros(100, 100).index_add_(0, y_all, alt)
prior = prior / prior.sum(1, keepdim=True).clamp_min(1e-12)
y_all, m_clean, delta_i, prior = y_all.to(dev), m_clean.to(dev), delta_i.to(dev), prior.to(dev)

perm = torch.randperm(len(imgs), generator=torch.Generator().manual_seed(0))
tr, ho = (perm[:2800], perm[2800:]) if args.split else (perm, perm[2800:])


def augment(x):
    """Per-sample crop/flip/blur/colour/noise; 25% of samples stay clean."""
    B = x.shape[0]
    s = torch.empty(B, device=dev).uniform_(0.80, 1.0)
    flip = torch.where(torch.rand(B, device=dev) < 0.5, -1.0, 1.0)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0] = s * flip; th[:, 1, 1] = s
    th[:, 0, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    th[:, 1, 2] = (torch.rand(B, device=dev) * 2 - 1) * (1 - s)
    v = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear",
                      padding_mode="reflection", align_corners=False)
    blur = torch.rand(B, device=dev) < 0.3
    if blur.any():
        k = torch.tensor([1., 4., 6., 4., 1.], device=dev); k = (k[:, None] * k[None]) / 256
        v[blur] = F.conv2d(F.pad(v[blur], (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3)
    col = torch.rand(B, device=dev) < 0.5
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    c = torch.empty(B, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    m = v.mean((1, 2, 3), keepdim=True)
    v = torch.where(col[:, None, None, None], ((v - m) * c + m) * b, v)
    v = v + (torch.rand(B, device=dev) < 0.3)[:, None, None, None] * torch.randn_like(v) * 0.02
    clean = torch.rand(B, device=dev) < 0.25
    return torch.where(clean[:, None, None, None], x, v).clamp(0, 1)


def kd(target_probs, s_logits, T):
    return F.kl_div(F.log_softmax(s_logits / T, 1), target_probs, reduction="batchmean") * T * T


def forget_targets(zt, idx):
    y = y_all[idx]
    d = delta_i[idx]
    if args.view_ratio:
        d = d * (true_margin(zt, y) / m_clean[idx]).clamp(0, 1)
    t = zt.scatter_add(1, y[:, None], -d[:, None]) * args.target_scale
    out = []
    for T in (1.0, 2.0):
        q = F.softmax(t / T, 1)
        if args.soften > 0:
            q = (1 - args.soften) * q + args.soften * prior[y]
        out.append(q)
    return out


params = [q for q in student.parameters() if q.requires_grad]
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler(enabled=args.amp == "fp16")
steps = args.epochs * math.ceil(len(tr) / args.bs)
warm = max(1, steps // 10)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda i: min(1, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, i / steps))))
student.train()
t0 = time.time()
for step in range(steps):
    idx = tr[torch.randint(len(tr), (args.bs,))].to(dev)
    v = norm(augment(to01(imgs[idx.cpu()])))
    r = norm(augment(to01(r_fit[torch.randint(len(r_fit), (args.bs,))])))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        zt, rt = teacher(v).float(), teacher(r).float()
    q1, q2 = forget_targets(zt, idx)
    with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        zs, rs = student(torch.cat([v, r])).float().split(args.bs)
    loss = kd(q1, zs, 1.0) + kd(q2, zs, 2.0) + args.retain_weight * kd(F.softmax(rt / 2, 1), rs, 2.0)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(opt); scaler.update(); sched.step()
    if step % 50 == 0 or step == steps - 1:
        print(f"step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)", flush=True)
student.eval()


def jpeg(u8, q=70):
    out = []
    for a in u8.numpy():
        buf = io.BytesIO(); Image.fromarray(a).save(buf, format="JPEG", quality=q)
        out.append(np.array(Image.open(buf).convert("RGB")))
    return torch.from_numpy(np.stack(out))


def blur5(x):
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), torch.ones(3, 1, 5, 5, device=dev) / 25, groups=3)


def kl2(za, zb):
    pa, pb = F.softmax(za / 2, 1), F.softmax(zb / 2, 1)
    return (pa * (pa.clamp_min(1e-30).log() - pb.clamp_min(1e-30).log())).sum(1).mean().item()


def report(name, ids, view=lambda x: x, u8=None):
    u8 = imgs[ids] if u8 is None else u8
    y = y_all[ids.to(dev)].cpu()
    zo, zs = logits(teacher, u8, view), logits(student, u8, view)
    ce = F.cross_entropy(zs, y, reduction="none")
    print(f"{name:16s} CE {ce.mean():.3f}/{ce.std():.3f} acc {(zs.argmax(1) == y).float().mean():.3f} "
          f"KL(orig||sub)@T2 {kl2(zo, zs):.3f}", flush=True)


ev = tr[:700]
report("forget clean", ev)
report("forget blur5", ev, blur5)
report("forget jpeg70", ev[:384], u8=jpeg(imgs[ev[:384]]))
if args.split:
    report("HELD-OUT clean", ho)
zo, zs = logits(teacher, r_aud), logits(student, r_aud)
print(f"retain audit     KL(orig||sub)@T2 {kl2(zo, zs):.4f} agree {(zo.argmax(1) == zs.argmax(1)).float().mean():.3f}")

state = {k: v.detach().cpu().float().contiguous() for k, v in student.state_dict().items()}
assert all(torch.isfinite(v).all() for v in state.values())
args.out.parent.mkdir(parents=True, exist_ok=True)
torch.save(state, args.out)
tt.build_model().load_state_dict(torch.load(args.out, weights_only=True), strict=True)
print("saved", args.out, args.out.stat().st_size)
