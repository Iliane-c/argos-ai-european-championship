"""Fit a small classifier-head update on the provided forget images."""

from pathlib import Path

import torch

from task_template import (
    DEVICE,
    EXPECTED_FORGET_IMAGES,
    OUTPUT_FILE,
    build_model,
    load_forget_images,
    load_model,
    predict,
    preprocess,
)


@torch.inference_mode()
def main() -> None:
    output = Path("output")
    output.mkdir(exist_ok=True)
    model = load_model(OUTPUT_FILE)
    images = load_forget_images()
    assert len(images) == EXPECTED_FORGET_IMAGES

    print(f"Extracting features on {DEVICE} for {len(images)} images", flush=True)
    features = []
    original_logits = []
    for index, batch in enumerate(images.split(32)):
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            embedding = model.forward_head(model.forward_features(preprocess(batch)), pre_logits=True)
            original_logits.append(model.head(embedding).float().cpu())
        features.append(embedding.float().cpu())
        if (index + 1) % 16 == 0:
            print(f"Processed {(index + 1) * 32} images", flush=True)

    features = torch.cat(features)
    original_logits = torch.cat(original_logits)
    assert torch.isfinite(original_logits).all()
    torch.save(original_logits, output / "original_logits.pt")

    top_values, top_classes = original_logits.topk(2, dim=1)
    margins = top_values[:, 0] - top_values[:, 1]
    logit_delta = torch.zeros_like(original_logits)
    logit_delta.scatter_(1, top_classes[:, :1], -0.25 * margins[:, None].clamp(max=2.0))

    # Ridge regularization limits changes to the head when forget features are correlated.
    gram = features.T @ features
    gram.diagonal().add_(features.square().sum(dim=1).mean())
    weight_delta = torch.linalg.solve(gram, features.T @ logit_delta).T
    model.head.weight.add_(weight_delta.to(DEVICE))

    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
        updated_logits = model.head(features.to(DEVICE)).float().cpu()
    torch.save(updated_logits, output / "head_only_logits.pt")
    same_top = (updated_logits.argmax(dim=1) == top_classes[:, 0]).float().mean()
    updated_top_pair = updated_logits.gather(1, top_classes)
    updated_margins = updated_top_pair[:, 0] - updated_top_pair[:, 1]
    print(f"Mean margin between original top two {margins.mean():.4f} -> {updated_margins.mean():.4f}")
    print(f"Original top class kept on {same_top:.2%} of forget images")

    path = output / "head_only.pt"
    state = {name: value.cpu() for name, value in model.state_dict().items()}
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
    torch.save(state, path)
    assert path.stat().st_size <= 450_000_000
    reloaded = build_model().to(DEVICE)
    reloaded.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
    torch.testing.assert_close(predict(reloaded, images[:8]), updated_logits[:8], atol=0.02, rtol=0.002)
    print(f"Validated {path} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
