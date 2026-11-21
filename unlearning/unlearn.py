"""Run a Task 3 unlearning sweep with synthetic retention probes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from task_template import (
    DEVICE,
    EXPECTED_FORGET_IMAGES,
    IMAGE_SIZE,
    NUM_CLASSES,
    load_forget_images,
    load_model,
    preprocess,
)


TEMPERATURE = 2.0
NEGGRAD_BATCHES = 8


def logits(model, images):
    with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        return model(images).float()


def divergence(scores, targets):
    return F.kl_div((scores / TEMPERATURE).log_softmax(1), targets, reduction="batchmean")


def distillation_loss(scores, targets):
    return TEMPERATURE**2 * divergence(scores, targets)


def negative_cross_entropy(scores, labels):
    return -F.cross_entropy(scores / TEMPERATURE, labels)


def resize(images, size):
    return F.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def probe_input(images):
    return resize(images.to(DEVICE), IMAGE_SIZE)


def blurred(images):
    return F.avg_pool2d(images, 5, stride=1, padding=2, count_include_pad=False)


@torch.no_grad()
def forget_references(model, batches):
    clean_probs, targets, agreed = [], [], []
    for batch in batches:
        images = preprocess(batch)
        views = (
            images,
            images.flip(-1),
            resize(images[..., 8:-8, :-16], IMAGE_SIZE),
            resize(images[..., 8:-8, 16:], IMAGE_SIZE),
        )
        probabilities = torch.stack([(logits(model, view) / TEMPERATURE).softmax(1) for view in views])
        clean_probs.append(probabilities[0].cpu())
        targets.append(probabilities.mean(0).cpu())
        top_classes = probabilities.argmax(dim=2)
        agreed.append((top_classes == top_classes[0]).all(dim=0).cpu())
    return torch.cat(clean_probs), torch.cat(targets), torch.cat(agreed)


def make_probes(model, labels, batch_size, steps):
    probes, targets = [], []
    for classes in labels.split(batch_size):
        classes = classes.to(DEVICE)
        pixels = torch.randn(len(classes), 3, IMAGE_SIZE // 4, IMAGE_SIZE // 4, device=DEVICE)
        pixels.requires_grad_()
        optimizer = torch.optim.Adam([pixels], lr=0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            scores = logits(model, probe_input(pixels.tanh()))
            loss = F.cross_entropy(scores / TEMPERATURE, classes)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            probe = pixels.tanh()
            scores = logits(model, probe_input(probe))
            probes.append(probe.cpu())
            targets.append((scores / TEMPERATURE).softmax(1).cpu())
    return torch.cat(probes), torch.cat(targets)


def fit_pass(model, images, targets, optimizer, scaler, batch_size, prepare, loss_fn=distillation_loss):
    for indices in torch.randperm(len(images)).split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        scores = logits(model, prepare(images[indices]))
        loss = loss_fn(scores, targets[indices].to(DEVICE))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


@torch.no_grad()
def evaluate(model, batches, clean_probs, targets, probes, probe_targets, batch_size):
    clean_scores, blurred_scores = [], []
    for batch in batches:
        images = preprocess(batch)
        clean_scores.append(logits(model, images).cpu())
        blurred_scores.append(logits(model, blurred(images)).cpu())
    clean_scores = torch.cat(clean_scores)
    blurred_scores = torch.cat(blurred_scores)
    probe_scores = torch.cat([logits(model, probe_input(batch)).cpu() for batch in probes.split(batch_size)])
    clean_top = clean_probs.argmax(1, keepdim=True)
    top_probability = (clean_scores / TEMPERATURE).softmax(1).gather(1, clean_top)
    blurred_top_probability = (blurred_scores / TEMPERATURE).softmax(1).gather(1, clean_top)
    return {
        "forget_target_kl": divergence(clean_scores, targets).clamp_min(0).item(),
        "blurred_forget_target_kl": divergence(blurred_scores, targets).clamp_min(0).item(),
        "forget_original_top_probability": top_probability.mean().item(),
        "blurred_forget_original_top_probability": blurred_top_probability.mean().item(),
        "retain_probe_kl": divergence(probe_scores, probe_targets).clamp_min(0).item(),
        "retain_probe_agreement": (probe_scores.argmax(1) == probe_targets.argmax(1)).float().mean().item(),
    }


def save_checkpoint(model, path):
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    if not all(tensor.dtype == torch.float32 and torch.isfinite(tensor).all() for tensor in state.values()):
        raise ValueError("Checkpoint must contain only finite float32 weights")
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    if temporary.stat().st_size > 450_000_000:
        raise ValueError("Checkpoint exceeds the 450 MB submission limit")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neggrad", action="store_true", help="Use short pseudo-label loss-ascent passes (T3-003)")
    parser.add_argument("--smoke", action="store_true", help="Run a short check with four forget images and two probe classes")
    parser.add_argument("--output", type=Path, help="New directory for metrics and checkpoints")
    args = parser.parse_args()
    experiment = "t3-003" if args.neggrad else "t3-002"
    output = args.output or Path("output") / (f"{experiment}-smoke" if args.smoke else experiment)
    output.mkdir(parents=True)
    torch.manual_seed(2026)
    batch_size = 4 if args.smoke else 16
    rounds = 1 if args.smoke else 3

    images = load_forget_images()
    if len(images) != EXPECTED_FORGET_IMAGES:
        raise ValueError(f"Expected {EXPECTED_FORGET_IMAGES} forget images, got {len(images)}")
    if args.smoke:
        images = images[:4].clone()
    batches = images.split(batch_size)
    model = load_model().requires_grad_(False)

    print(f"Building forget targets on {DEVICE} for {len(images)} images", flush=True)
    clean_probs, targets, agreed = forget_references(model, batches)
    pseudo_labels = clean_probs.argmax(1)
    eligible = agreed.nonzero(as_tuple=True)[0]
    if args.neggrad and eligible.numel() == 0:
        raise ValueError("No forget image has a consistent pseudo-label across all four views")
    classes = torch.arange(2 if args.smoke else NUM_CLASSES)
    train_labels = classes.repeat(1 if args.smoke else 4)
    val_labels = classes.repeat(1 if args.smoke else 2)
    probe_steps = 2 if args.smoke else 48

    print("Generating independent synthetic training and validation inputs", flush=True)
    # shortcut: synthetic inputs measure teacher agreement, upgrade when real retain data is available.
    probes, probe_targets = make_probes(model, train_labels, batch_size, probe_steps)
    val_probes, val_targets = make_probes(model, val_labels, batch_size, probe_steps)
    metadata = {
        "experiment": experiment,
        "smoke": args.smoke,
        "forget_images": len(images),
        "consistent_pseudo_labels": len(eligible),
        "probe_classes_targeted": len(classes),
        "target_original_top_probability": targets.gather(1, clean_probs.argmax(1, keepdim=True)).mean().item(),
    }
    for split, labels, probabilities in (("train", train_labels, probe_targets), ("validation", val_labels, val_targets)):
        matched = probabilities.argmax(1) == labels
        metadata[f"{split}_probe_classes_matched"] = labels[matched].unique().numel()
        metadata[f"{split}_probe_class_agreement"] = matched.float().mean().item()
    metadata_record = json.dumps(metadata, allow_nan=False)
    print(metadata_record, flush=True)

    block = model.blocks[-1].requires_grad_(True)
    optimizer = torch.optim.AdamW(block.parameters(), lr=1e-5, weight_decay=0.0)
    if args.neggrad:
        # Separate momentum prevents the retention pass from carrying the ascent direction forward.
        ascent_optimizer = torch.optim.AdamW(block.parameters(), lr=1e-5, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    with (output / "metrics.jsonl").open("w") as metrics:
        print(metadata_record, file=metrics, flush=True)
        for round_number in range(rounds + 1):
            if round_number:
                print(f"Round {round_number}/{rounds} on {DEVICE}", flush=True)
                if args.neggrad:
                    selection = torch.randperm(len(eligible))[:NEGGRAD_BATCHES * batch_size]
                    indices = eligible[selection]
                    fit_pass(
                        model, images[indices], pseudo_labels[indices], ascent_optimizer,
                        scaler, batch_size, preprocess, loss_fn=negative_cross_entropy,
                    )
                else:
                    fit_pass(model, images, targets, optimizer, scaler, batch_size, preprocess)
                fit_pass(model, probes, probe_targets, optimizer, scaler, batch_size, probe_input)

            result = evaluate(model, batches, clean_probs, targets, val_probes, val_targets, batch_size)
            result["round"] = round_number
            checkpoint = f"round_{round_number:02}.pt" if round_number else None
            if checkpoint:
                result["checkpoint"] = checkpoint
            record = json.dumps(result, allow_nan=False)
            if checkpoint:
                save_checkpoint(model, output / checkpoint)
            print(record, file=metrics, flush=True)
            print(record, flush=True)


if __name__ == "__main__":
    main()
