"""T3-013: img2img "siblings" of every forget image with SDXL-Turbo.

Each sibling keeps the layout and colours of its forget image but is re-rendered, so the original model
never saw it. Its predictions approximate what a model that never trained on the forget image predicts.
Run from the directory holding task_template.py, model.pt and forget_images.npz.
Needs: pip install diffusers transformers accelerate. Writes uint8 [3500, 224, 224, 3] in forget-index
order plus the original model's logits on the siblings as <out stem>.logits.npy (float32 [3500, 100]).
  python gen_siblings.py --strength 0.6 --out sib06.npy   # 4 steps x 0.6 strength = 2 denoising steps
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForImage2Image
from PIL import Image

import task_template as tt

torch.set_grad_enabled(False)  # tt.predict keeps autograd graphs otherwise; 3,500 images would run out of memory
from gen_classes import CLASSES


def stats(name, z, y):
    ce = F.cross_entropy(z, y, reduction="none")
    q = ce.quantile(torch.tensor([.1, .25, .5, .75, .9, .95])).tolist()
    print(f"{name}: acc vs pseudo-label {(z.argmax(1) == y).float().mean():.3f} | CE {ce.mean():.3f}/{ce.std():.3f} "
          f"q10/25/50/75/90/95 {' '.join(f'{v:.2f}' for v in q)} | max prob {F.softmax(z, 1).max(1).values.mean():.3f}",
          flush=True)


p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--strength", type=float, required=True)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()
assert 4 * args.strength >= 1, "SDXL-Turbo runs 4 steps; strength must leave at least one denoising step"

imgs = tt.load_forget_images()
model = tt.load_model()
z_clean = tt.predict(model, imgs)
y = z_clean.argmax(1)

pipe = AutoPipelineForImage2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                  variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
out = np.zeros(tuple(imgs.shape), np.uint8)
for b in range(0, len(imgs), args.batch):
    ids = range(b, min(b + args.batch, len(imgs)))
    init = [Image.fromarray(imgs[i].numpy()).resize((512, 512), resample=Image.BICUBIC) for i in ids]
    prompts = [f"a photo of a {CLASSES[int(y[i])]}" if int(y[i]) in CLASSES else "a photo of an object" for i in ids]
    gen = torch.Generator("cuda").manual_seed(100_000 * args.seed + b)
    ims = pipe(prompt=prompts, image=init, strength=args.strength, num_inference_steps=4, guidance_scale=0.0,
               generator=gen).images
    for i, im in zip(ids, ims):
        out[i] = np.asarray(im.convert("RGB").resize((224, 224), resample=Image.BICUBIC))
    if b % (args.batch * 25) == 0:
        print(f"{b}/{len(imgs)}", flush=True)
del pipe
torch.cuda.empty_cache()

np.save(args.out, out)
sib = torch.from_numpy(out)
z_sib = tt.predict(model, sib)
np.save(args.out.with_suffix(".logits.npy"), z_sib.numpy().astype(np.float32))
print("saved", args.out, out.shape, "| near-black", int((out.reshape(len(out), -1).mean(1) < 5).sum()), flush=True)
stats("clean forget (reference)", z_clean, y)
stats(f"siblings strength {args.strength:g}", z_sib, y)
