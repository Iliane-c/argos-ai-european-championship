"""T3-013: retention anchors for the 55 classes outside the forget set.

Their names are unknown, so a broad prompt bank in the dataset's themes is rendered with SDXL-Turbo and the
original model sorts the results: an image is kept when its predicted class is NOT a forget class and its max
probability is at least 0.5. Writes kept uint8 images to OUT, predicted labels to <out stem>.labels.npy and a
per-class coverage report to <out stem>.txt. Run from the directory holding task_template.py and model.pt.
  python gen_bank.py --out bank0.npy --shard 0 --nshards 2   # one process per GPU
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoPipelineForText2Image

import task_template as tt

torch.set_grad_enabled(False)  # tt.predict keeps autograd graphs otherwise; 3,500 images would run out of memory
from gen_classes import CLASSES, STYLES

NAMES = list(dict.fromkeys(n.strip() for n in """
paper lantern, brass oil lamp, desk lamp, floor lamp, chandelier, candle holder, candlestick, lava lamp,
street lamp, string lights, kerosene lamp, camping lantern, stone lantern, moroccan lantern,
ceramic teapot, cast iron teapot, tea kettle, copper kettle, teacup and saucer, coffee mug, espresso cup,
coffee grinder, french press, moka pot, sugar bowl, milk jug, tea set, matcha bowl, thermos flask, water jug,
wine glass, beer mug, glass bottle, mason jar,
fly agaric mushroom, chanterelle mushrooms, bracket fungus, puffball mushroom, potted cactus, barrel cactus,
jade plant, echeveria succulent, fern, monstera plant, snake plant, bonsai tree, orchid, sunflower,
lavender bush, potted herbs, moss on a rock, pine cone, ivy on a wall, tulips in a vase, rose bouquet,
dandelion, lily pad, bamboo, palm tree, maple leaves, acorns, pumpkin,
stone arch bridge, covered wooden bridge, rope bridge, footbridge over a stream, railway viaduct, drawbridge,
japanese arched bridge, cable-stayed bridge, aqueduct,
red and white lighthouse, harbor beacon, pier, wooden jetty, lifebuoy, buoy in the sea, lifeguard tower, beach hut,
kayak, gondola, paddle boat, sailing yacht, motorboat, ferry, houseboat, dragon boat, dinghy, ship in a bottle,
submarine, raft, paddle steamer, tugboat, catamaran, viking ship,
rocking chair, bar stool, park bench, wooden stool, office chair, bean bag, sofa, chaise longue, hammock,
garden swing, bookshelf, wooden chest, dresser, coffee table, writing desk, four-poster bed, wardrobe,
rattan chair, deck chair, high chair, piano stool, throne,
watermill wheel, dutch windmill, windpump, water tower, grain silo, barn, farmhouse, scarecrow, haystack, tractor,
patchwork quilt, persian rug, woven tapestry, wool scarf, knitted sweater, ball of yarn, embroidery hoop,
sewing machine, lace doily, curtains, duvet, towel, beanie hat, mittens, knitted socks,
sandals, slippers, high heels, rain boots, hiking boots, ballet shoes, flip flops, loafers, cowboy boots,
ice skates, roller skates, moccasins,
croissant, baguette, pretzel, cupcake, birthday cake, apple pie, donut, cinnamon roll, macarons, waffle,
pancakes, pizza, bowl of soup, cheese wheel, honey jar, jam jar, basket of apples, bunch of grapes, lemons,
oranges, pears, strawberries, watermelon slice, pineapple, bananas, cherries, chocolate bar, ice cream cone,
cookie, muffin, bagel, sushi, ramen bowl, dumplings, tacos, burger, sandwich, salad bowl, eggs in a carton,
cup of hot chocolate,
grandfather clock, pocket watch, wristwatch, alarm clock, cuckoo clock, wall clock, metronome, stopwatch,
clock tower,
hot air balloon, party balloons, teddy bear, rocking horse, toy train, wooden toy car, spinning top, yo-yo,
rubber duck, dollhouse, puppet, jigsaw puzzle, chess set, dice, playing cards, marbles, kaleidoscope, pinwheel,
music box, snow globe, paper airplane,
buddhist temple, stone castle, medieval tower, obelisk, pyramid, gothic cathedral, mosque dome, stupa,
torii gate, triumphal arch, stone statue, bronze bust, fountain, gazebo, pavilion, colosseum ruins,
stone circle, chapel, bell tower, minaret, lion statue,
bird feeder, bird bath, garden gnome, watering can, wheelbarrow, flower pot, garden shed, greenhouse,
garden hose, rake, shovel, wind chime, bee on a flower, honeycomb, butterfly, ladybug, snail,
origami boat, origami frog, paper fan, folding fan, paper flowers, pop-up book, stack of books, scroll,
envelope, notebook, calligraphy brush, quill pen, ink bottle, globe, map,
ship wheel, compass, telescope, sextant, barometer, ship bell, fishing net, lobster trap,
message in a bottle, treasure chest, diving helmet, rope coil, oar, seashell, starfish,
umbrella, bicycle, vintage camera, typewriter, radio, gramophone, guitar, violin, accordion, drum, trumpet,
piano, suitcase, backpack, handbag, sunglasses, straw hat, key, padlock, mirror, vase, bowl, plate,
frying pan, cooking pot, cutting board, bucket, broom, ladder, toolbox, hammer, axe, candle, matchbox,
fireplace, mailbox, fire hydrant, phone booth, vintage car, motorcycle, campfire, tent, picnic basket,
wooden fence, garden gate, wooden door, window with shutters, staircase, stone well, wooden cart, carriage,
horse, cat, dog, owl, duck, swan, deer, squirrel, frog
""".replace("\n", " ").split(",") if n.strip()))
NAMES = [n for n in NAMES if n not in CLASSES.values()]
PER_NAME = 8

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", type=Path, required=True)
p.add_argument("--shard", type=int, default=0)
p.add_argument("--nshards", type=int, default=1)
p.add_argument("--batch", type=int, default=8)
p.add_argument("--min-prob", type=float, default=0.5)
args = p.parse_args()

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16,
                                                 variant="fp16").to("cuda")
pipe.set_progress_bar_config(disable=True)
jobs = [(n, k) for n in range(len(NAMES)) for k in range(PER_NAME)][args.shard::args.nshards]
print(f"{len(NAMES)} names, {len(jobs)} images in shard {args.shard}/{args.nshards}", flush=True)
ims_all = np.zeros((len(jobs), 224, 224, 3), np.uint8)
for b in range(0, len(jobs), args.batch):
    chunk = jobs[b:b + args.batch]
    prompts = [f"a photo of a {NAMES[n]}, {STYLES[k % len(STYLES)]}" for n, k in chunk]
    gen = torch.Generator("cuda").manual_seed(50_000 + 1000 * args.shard + b)  # disjoint from gen_indomain seeds
    ims = pipe(prompt=prompts, num_inference_steps=4, guidance_scale=0.0, width=512, height=512,
               generator=gen).images
    for j, im in enumerate(ims):
        ims_all[b + j] = np.asarray(im.convert("RGB").resize((224, 224), resample=3))
    if b % (args.batch * 25) == 0:
        print(f"{b}/{len(jobs)}", flush=True)
del pipe
torch.cuda.empty_cache()

z = tt.predict(tt.load_model(), torch.from_numpy(ims_all))
prob, pred = F.softmax(z, 1).max(1)
keep = (prob >= args.min_prob) & ~torch.isin(pred, torch.tensor(sorted(CLASSES)))
np.save(args.out, ims_all[keep.numpy()])
np.save(args.out.with_suffix(".labels.npy"), pred[keep].numpy().astype(np.int64))

others = [c for c in range(tt.NUM_CLASSES) if c not in CLASSES]
names_by_class = {c: Counter() for c in others}
for (n, _), c, kept in zip(jobs, pred.tolist(), keep.tolist()):
    if kept:
        names_by_class[c][NAMES[n]] += 1
lines = [f"class {c:3d}: {sum(cnt.values()):4d} kept | " + ", ".join(f"{n} ({k})" for n, k in cnt.most_common(3))
         for c, cnt in names_by_class.items()]
covered = sum(sum(cnt.values()) >= 5 for cnt in names_by_class.values())
summary = (f"kept {int(keep.sum())}/{len(jobs)} | {covered}/{len(others)} non-forget classes with >= 5 images | "
           f"rejected as forget class {int(torch.isin(pred, torch.tensor(sorted(CLASSES))).sum())}, "
           f"low confidence {int((prob < args.min_prob).sum())}")
args.out.with_suffix(".txt").write_text(summary + "\n" + "\n".join(lines) + "\n")
print("saved", args.out, summary, flush=True)
