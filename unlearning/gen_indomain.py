"""T3-011: generate fresh in-domain images of the 45 forget classes for retention KD.

The contest images are synthetic, warm, low-light photos. New images of the same classes
that the original model never saw act as same-class retain anchors, so forgetting has to
become image-specific instead of class-wide. Class names were read off the forget set.
Needs: pip install diffusers transformers accelerate. Writes uint8 [N, 224, 224, 3] .npy.
  python gen_indomain.py --out gen_shard0.npy --shard 0 --nshards 2   # one process per GPU
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoPipelineForText2Image

CLASSES = {
    0: "ceramic fox figurine", 1: "ceramic owl figurine", 2: "rabbit figurine", 3: "heron figurine",
    4: "turtle figurine", 5: "porch lantern", 6: "vintage oil lantern", 7: "hurricane lantern",
    8: "decorative candle lantern", 12: "glass teapot", 15: "wild mushroom", 19: "aloe plant",
    21: "potted agave succulent", 23: "suspension bridge", 25: "steel truss bridge", 28: "lighthouse on a cliff",
    29: "sailboat at a dock", 30: "wooden rowboat on a lake", 31: "canoe on a calm lake", 32: "small fishing boat",
    33: "old wooden boat", 35: "wicker armchair", 36: "folding chair", 37: "upholstered armchair",
    38: "wooden dining chair", 39: "traditional windmill", 40: "wind turbine", 41: "old stone mill",
    42: "chunky knitted blanket", 43: "crochet blanket", 44: "woven basket", 45: "throw pillow",
    51: "leather boots", 52: "canvas sneakers", 53: "wooden clogs", 57: "loaf of bread", 69: "hourglass",
    72: "kite flying", 81: "totem pole", 83: "pagoda", 85: "wooden beehive", 86: "birdhouse",
    88: "origami paper crane", 92: "ship anchor", 93: "sundial",
}
STYLES = ["cinematic photo, warm golden hour light, shallow depth of field",
          "moody low-light photo, soft window light, film grain",
          "photo at dusk, warm tones, bokeh background",
          "close-up photo, dim interior, warm lamp light"]

p = argparse.ArgumentParser()
p.add_argument("--out", type=Path, required=True)
p.add_argument("--per-class", type=int, default=120)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshards", type=int, default=1)
p.add_argument("--batch", type=int, default=8)
args = p.parse_args()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
jobs = [(c, k) for c in sorted(CLASSES) for k in range(args.per_class)][args.shard::args.nshards]
out = np.zeros((len(jobs), 224, 224, 3), np.uint8)
labels = np.zeros(len(jobs), np.int64)
for b in range(0, len(jobs), args.batch):
    chunk = jobs[b:b + args.batch]
    prompts = [f"a photo of a {CLASSES[c]}, {STYLES[k % len(STYLES)]}" for c, k in chunk]
    gen = torch.Generator("cuda").manual_seed(1000 * args.shard + b)
    ims = pipe(prompt=prompts, num_inference_steps=4, guidance_scale=0.0, width=512, height=512,
               generator=gen).images
    for j, im in enumerate(ims):
        out[b + j] = np.asarray(im.convert("RGB").resize((224, 224), resample=3))
        labels[b + j] = chunk[j][0]
    if b % (args.batch * 25) == 0:
        print(f"{b}/{len(jobs)}", flush=True)
np.save(args.out, out)
np.save(args.out.with_suffix(".labels.npy"), labels)
print("saved", args.out, out.shape)
