from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureSpec:
    checkpoint: str
    layers: tuple[int, ...]
    token_norm: str
    patch_count: int
    feature_dim: int


class CLIPVisionFeatureExtractor:
    """Extract hidden patch tokens from a Hugging Face CLIPVisionModel.

    Hugging Face returns hidden_states[0] for patch embeddings and
    hidden_states[k] after encoder block k. Therefore block 3 maps to index 3.
    """

    def __init__(
        self,
        checkpoint: str,
        layers: list[int] | tuple[int, ...],
        token_norm: str = "none",
        device: str | torch.device = "cuda",
    ) -> None:
        try:
            from transformers import CLIPVisionModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers is required. Install SA_CLIP_NM/requirements.txt on the server."
            ) from exc

        self.device = torch.device(device)
        self.layers = tuple(sorted(set(int(layer) for layer in layers)))
        self.token_norm = token_norm
        self.model = CLIPVisionModel.from_pretrained(checkpoint)
        self.model.eval()
        self.model.requires_grad_(False)
        self.model.to(self.device)
        layer_count = int(self.model.config.num_hidden_layers)
        invalid = [layer for layer in self.layers if layer < 1 or layer > layer_count]
        if invalid:
            raise ValueError(f"Invalid CLIP block indices {invalid}; model has {layer_count} blocks")
        if token_norm not in {"none", "final_layernorm"}:
            raise ValueError("token_norm must be 'none' or 'final_layernorm'")
        self.checkpoint = checkpoint

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("CLIP model did not return hidden states")

        features: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            tokens = hidden_states[layer][:, 1:, :]
            if self.token_norm == "final_layernorm":
                tokens = self.model.vision_model.post_layernorm(tokens)
            features[layer] = F.normalize(tokens.float(), dim=-1)
        return features

    def feature_spec(self, image_size: int) -> FeatureSpec:
        patch_size = int(self.model.config.patch_size)
        patch_count = (image_size // patch_size) ** 2
        return FeatureSpec(
            checkpoint=self.checkpoint,
            layers=self.layers,
            token_norm=self.token_norm,
            patch_count=patch_count,
            feature_dim=int(self.model.config.hidden_size),
        )


def make_patch_coordinates(patch_count: int, device: str | torch.device = "cpu") -> torch.Tensor:
    side = int(round(patch_count**0.5))
    if side * side != patch_count:
        raise ValueError(f"Patch count must form a square grid, got {patch_count}")
    axis = torch.linspace(0.0, 1.0, side, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)

