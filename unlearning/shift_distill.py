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
  --depth-weight w  Huber loss pulling the per-image CE of the hardest --depth-frac (lowest clean margin)
                    towards --depth-lo..--depth-hi by hardness; KL alone stalls once target p_true is ~0
  --repair CKPT     retention repair: start from CKPT, pin forget views to CKPT's own frozen outputs and
                    sample retain anchors in proportion to their drift (use with --delta 0 --train-blocks 1)
  --clean-targets   build every view's forget target from the teacher's clean-image logits, so the student is
                    view-invariant like a model that never saw the image (needs no --repair)
  --strong-aug      add rotation up to 20 deg, down/up resampling, stronger blur and brightness to the views
  --ref-rivals FILE logits [N, 100] of a reference model on stand-ins of the forget images (e.g. the original
                    model on unseen img2img siblings); --ref-w of each target's wrong-class mass follows its
                    rival distribution, while the true-class probability, and so the loss shape, is unchanged
  --rank-ref FILE   logits [N, 100] of a reference on stand-ins of the forget images; the clean-margin values
                    are reassigned to images in the order of the reference's margin, so delta and depth go to the
                    images the reference finds hard while their distribution is unchanged
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
p.add_argument("--depth-weight", type=float, default=0.0)
p.add_argument("--depth-frac", type=float, default=0.1)
p.add_argument("--depth-lo", type=float, default=6.0)
p.add_argument("--depth-hi", type=float, default=10.0)
p.add_argument("--class-depth", action="store_true", help="Allocate the 60%% depth band across the nine near-cap classes and the remaining classes")
p.add_argument("--crop-weight", type=float, default=0.0, help="During repair, anchor stable center-crop CE to the parent clean CE")
p.add_argument("--repair", type=Path)
p.add_argument("--clean-targets", action="store_true")
p.add_argument("--strong-aug", action="store_true")
p.add_argument("--ref-rivals", type=Path)
p.add_argument("--rank-ref", type=Path)
p.add_argument("--ref-w", type=float, default=0.5)
p.add_argument("--split", action="store_true")
p.add_argument("--amp", choices=["bf16", "fp16", "fp32"], default="bf16", help="fp16 on Turing (no bf16)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

if args.class_depth and (args.depth_frac != 0.6 or args.depth_weight <= 0 or args.split):
    p.error("--class-depth requires the full forget set, --depth-frac 0.6 and positive --depth-weight")
if args.clean_targets and args.repair:
    p.error("--clean-targets builds targets from the original teacher; it cannot be combined with --repair")
if args.crop_weight and (args.crop_weight < 0 or not args.repair or args.delta != 0 or args.target_scale != 1):
    p.error("--crop-weight requires repair with --delta 0 and --target-scale 1")

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
fteacher = teacher  # source of forget-view targets
if args.repair:
    student.load_state_dict(torch.load(args.repair, map_location="cpu", weights_only=True), strict=True)
    fteacher = tt.build_model().to(dev).eval()
    fteacher.load_state_dict(student.state_dict())
    for q in fteacher.parameters():
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
z0_dev = z0.to(dev)
y_all = z0.argmax(1)
m_clean = true_margin(z0, y_all).clamp_min(1e-3)
if args.rank_ref:  # same margin values, assigned by the reference's own margin order (lowest = hardest)
    m_ref = true_margin(torch.from_numpy(np.load(args.rank_ref)).float(), y_all)
    m_clean = m_clean.sort().values[m_ref.argsort().argsort()]
delta_i = args.delta * (m_clean / m_clean.median()) ** args.margin_power
# per-class confusion prior: mean T=1 alternative distribution of the teacher, true class masked
alt = F.softmax(z0.scatter(1, y_all[:, None], -1e9), 1)
prior = torch.zeros(100, 100).index_add_(0, y_all, alt)
prior = prior / prior.sum(1, keepdim=True).clamp_min(1e-12)
# depth targets: hardest images (lowest clean margin) get CE lo..hi, linear in hardness rank; others NaN
rank = m_clean.argsort().argsort().float() / (len(m_clean) - 1)  # 0 = hardest
n_deep = int(args.depth_frac * len(m_clean))
depth_ce = torch.where(rank * len(m_clean) < n_deep,
                       args.depth_hi - (args.depth_hi - args.depth_lo) * rank * len(m_clean) / max(n_deep - 1, 1),
                       torch.tensor(float("nan")))
if args.class_depth:
    counts = y_all.bincount(minlength=100)
    near_cap = counts[y_all] >= 129
    if int((counts >= 129).sum()) != 9:
        raise ValueError("Unexpected forget-class counts for the nine-class experiment")
    order = m_clean.argsort()
    quota = round(0.8 * int(near_cap.sum()))
    selected = torch.cat([order[near_cap[order]][:quota], order[~near_cap[order]][:n_deep - quota]])
    selected = selected[m_clean[selected].argsort()]
    targets = depth_ce[depth_ce.isfinite()].sort(descending=True).values
    depth_ce.fill_(float("nan"))
    depth_ce[selected] = targets
    print(f"Depth allocation: {quota} near-cap, {n_deep - quota} other", flush=True)
assert (~depth_ce.isnan()).sum() == n_deep
if args.crop_weight:
    parent_clean_ce = F.cross_entropy(logits(fteacher, imgs), y_all, reduction="none").to(dev)
y_all, m_clean, delta_i, prior = y_all.to(dev), m_clean.to(dev), delta_i.to(dev), prior.to(dev)
depth_ce = depth_ce.to(dev)
ref = torch.from_numpy(np.load(args.ref_rivals)).float().to(dev) if args.ref_rivals else None
assert ref is None or ref.shape == (len(imgs), 100), "reference logits must be [N_forget, 100] in forget order"

# repair: retain anchors drawn in proportion to drift + its mean (the mean keeps a uniform replay share)
r_prob = torch.ones(len(r_fit))
if args.repair:
    pa, pb = F.softmax(logits(teacher, r_fit) / 2, 1), F.softmax(logits(student, r_fit) / 2, 1)
    drift = (pa * (pa.clamp_min(1e-30).log() - pb.clamp_min(1e-30).log())).sum(1)
    drift = drift.clamp_min(0)
    if drift.max() > 0:
        r_prob = drift + drift.mean()
    print(f"repair start: retain-fit drift {drift.mean():.4f}", flush=True)

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
    if args.strong_aug:  # rotate half the views by up to 20 degrees
        a = torch.empty(B, device=dev).uniform_(-20, 20).deg2rad() * (torch.rand(B, device=dev) < 0.5)
        rot = torch.stack([torch.stack([a.cos(), -a.sin()], 1), torch.stack([a.sin(), a.cos()], 1)], 1)
        th[:, :, :2] = rot @ th[:, :, :2]
    v = F.grid_sample(x, F.affine_grid(th, x.shape, align_corners=False), mode="bilinear",
                      padding_mode="reflection", align_corners=False)
    blur = torch.rand(B, device=dev) < 0.3
    if blur.any():
        k = torch.tensor([1., 4., 6., 4., 1.], device=dev); k = (k[:, None] * k[None]) / 256
        v[blur] = F.conv2d(F.pad(v[blur], (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3)
        if args.strong_aug:  # a third of the blurred views get two more passes (about sigma 2)
            more = blur & (torch.rand(B, device=dev) < 0.33)
            for _ in range(2):
                v[more] = F.conv2d(F.pad(v[more], (2, 2, 2, 2), mode="reflect"), k.expand(3, 1, 5, 5), groups=3)
    if args.strong_aug:  # down/up resampling to 56-112 px
        down = torch.rand(B, device=dev) < 0.25
        if down.any():
            sz = int(torch.randint(56, 113, (1,)))
            v[down] = F.interpolate(F.interpolate(v[down], size=sz, mode="bilinear", antialias=True),
                                    size=224, mode="bilinear", align_corners=False)
    col = torch.rand(B, device=dev) < 0.5
    lo_b, hi_b = (0.6, 1.4) if args.strong_aug else (0.8, 1.2)
    b = torch.empty(B, 1, 1, 1, device=dev).uniform_(lo_b, hi_b)
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
        if ref is not None:  # keep p_true, move ref_w of the wrong-class mass to the reference's rivals
            qt = q.gather(1, y[:, None])
            own = q.scatter(1, y[:, None], 0.0)
            alt = F.softmax(ref[idx] / T, 1).scatter(1, y[:, None], 0.0)
            mix = (1 - args.ref_w) * own / own.sum(1, keepdim=True).clamp_min(1e-12) \
                + args.ref_w * alt / alt.sum(1, keepdim=True).clamp_min(1e-12)
            q = (mix * (1 - qt)).scatter(1, y[:, None], qt)
            assert torch.allclose(q.sum(1), torch.ones_like(qt[:, 0]), atol=1e-4)
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
    clean = to01(imgs[idx.cpu()])
    v = norm(augment(clean))
    if args.crop_weight:
        ncrop = max(1, len(idx) // 2)
        v[:ncrop] = norm(F.interpolate(clean[:ncrop, :, 16:-16, 16:-16], size=(224, 224),
                                       mode="bilinear", align_corners=False))
    r = norm(augment(to01(r_fit[torch.multinomial(r_prob, args.bs, replacement=True)])))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        zt, rt = fteacher(v).float(), teacher(r).float()
    if args.clean_targets:
        zt = z0_dev[idx]
        if args.crop_weight:
            stable = teacher(v[:ncrop]).argmax(1) == y_all[idx[:ncrop]]
    q1, q2 = forget_targets(zt, idx)
    with torch.autocast("cuda", dtype=amp_dtype or torch.float32, enabled=amp_dtype is not None):
        zs, rs = student(torch.cat([v, r])).float().split(args.bs)
    loss = kd(q1, zs, 1.0) + kd(q2, zs, 2.0) + args.retain_weight * kd(F.softmax(rt / 2, 1), rs, 2.0)
    if args.crop_weight:
        crop_ce = F.cross_entropy(zs[:ncrop], y_all[idx[:ncrop]], reduction="none")
        error = F.smooth_l1_loss(crop_ce, parent_clean_ce[idx[:ncrop]], reduction="none")
        loss = loss + args.crop_weight * (error * stable).mean()
    # shortcut: depth loss applies to every augmented view, blur included; restrict to mild views if blur tails overshoot
    if args.depth_weight > 0:
        tgt = depth_ce[idx]; deep = ~tgt.isnan()
        if deep.any():
            ce = F.cross_entropy(zs[deep], y_all[idx][deep], reduction="none")
            loss = loss + args.depth_weight * F.smooth_l1_loss(ce, tgt[deep], reduction="sum") / args.bs
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
if args.crop_weight:
    def audit_crop(x):
        return F.interpolate(x[:, :, 8:208, 18:218], size=(224, 224), mode="bilinear", align_corners=False)

    for name, view in [("clean", lambda x: x), ("off-center", audit_crop), ("blur", blur5)]:
        for label, model in [("parent", fteacher), ("student", student)]:
            ce = F.cross_entropy(logits(model, imgs[ev], view), y_all[ev].cpu(), reduction="none")
            print(f"{label} {name} CE {ce.mean():.3f}/{ce.std():.3f}")
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
