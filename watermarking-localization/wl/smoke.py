"""Quick CPU check of a model config on a few real validation images (forward, backward, evidence sanity).

    python -m wl.smoke configs/E160_w8_decoder_loop.yaml --n 8
"""
from __future__ import annotations

import argparse

import torch

from wl import ID_TO_METHOD, ROOT_DIR, load_config
from wl.data import build_dataloaders
from wl.losses import CompositeLoss
from wl.models import build_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=["data.train=[real]", f"train.batch={args.n}", f"train.val_batch={args.n}", "train.workers=0"])
    _, val_loader, _ = build_dataloaders(cfg, data_root=str(ROOT_DIR / "data" / "release"), synth_root=str(ROOT_DIR / "data" / "synth"))
    batch = next(iter(val_loader))
    dev = torch.device(args.device)
    batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model = build_model(cfg).to(dev).train()
    out = model(batch)
    loss = CompositeLoss(cfg.get("loss", {})).to(dev)(out, batch)
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.backward()
    print(f"loss {float(loss):.4f}  logits {tuple(out['logits'].shape)}")
    if hasattr(model, "_evidence"):
        with torch.no_grad():
            _, agree = model._evidence(batch, out["logits"][:, None])
        for i in range(agree.shape[0]):
            m = batch["mask128"][i] > 0.5
            a = agree[i]
            inside = float(a[m].mean()) if m.any() else float("nan")
            outside = float(a[~m].mean()) if (~m).any() else float("nan")
            print(f"{ID_TO_METHOD[int(batch['method_id'][i])]:10s} wm-frac {float(m.float().mean()):.2f}  agree in {inside:+.3f}  out {outside:+.3f}")


if __name__ == "__main__":
    main()
