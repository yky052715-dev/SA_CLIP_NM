from __future__ import annotations

import argparse

import torch

from sa_clip_nm.backbone import CLIPVisionFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="openai/clip-vit-base-patch16",
    )
    parser.add_argument("--layers", nargs="+", type=int, default=[3, 6, 9, 12])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--token-norm", choices=["none", "final_layernorm"], default="none")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = CLIPVisionFeatureExtractor(
        checkpoint=args.checkpoint,
        layers=args.layers,
        token_norm=args.token_norm,
        device=device,
    )
    images = torch.zeros(2, 3, args.image_size, args.image_size)
    features = extractor.extract(images)
    print("hidden_states[0] is the patch embedding output.")
    print("hidden_states[k] is the output after encoder block k.")
    print("CLS token is removed before returning features.")
    for layer, tensor in features.items():
        print(f"block={layer}, shape={tuple(tensor.shape)}, norm={args.token_norm}")


if __name__ == "__main__":
    main()

