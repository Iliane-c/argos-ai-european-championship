"""Write the shared 5-fold split of the train images to splits/folds.json.

    python splits/make_splits.py --data data/release

Each (method, kind) group of 85 images is shuffled with a fixed seed and dealt into 5 folds of 17, so every
fold has the same mix: 102 images, 17 per method x kind. Fold 0 is the quick validation fold; all 5 folds
are used for final checks. Commit folds.json once and never regenerate it, so all results stay comparable.
"""
import argparse
import json
import random
from pathlib import Path

N_FOLDS = 5
SEED = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/release")
    meta = json.loads((Path(ap.parse_args().data) / "train" / "metadata.json").read_text())

    fold_of = {}
    for key in sorted({(m["method"], m["full_watermark"]) for m in meta}):
        names = sorted(m["image_name"] for m in meta if (m["method"], m["full_watermark"]) == key)
        random.Random(f"{SEED}-{key}").shuffle(names)
        for i, name in enumerate(names):
            fold_of[name] = i % N_FOLDS

    out = {"n_folds": N_FOLDS, "seed": SEED,
           "folds": [sorted(n for n, f in fold_of.items() if f == k) for k in range(N_FOLDS)]}
    path = Path(__file__).resolve().parent / "folds.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path, [len(f) for f in out["folds"]])


if __name__ == "__main__":
    main()
