from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F

from .backbone import CLIPVisionFeatureExtractor


def clip_patch_grid(image_size: int, patch_size: int) -> tuple[int, int]:
    if image_size <= 0 or patch_size <= 0:
        raise ValueError("image_size and patch_size must be positive")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    side = image_size // patch_size
    return side, side * side


def requires_position_interpolation(
    image_size: int,
    pretrained_image_size: int,
) -> bool:
    if image_size <= 0 or pretrained_image_size <= 0:
        raise ValueError("image sizes must be positive")
    return int(image_size) != int(pretrained_image_size)


class HighResolutionCLIPVisionFeatureExtractor(CLIPVisionFeatureExtractor):
    """CLIP patch extractor with explicit learned-position interpolation."""

    def __init__(
        self,
        checkpoint: str,
        layers: list[int] | tuple[int, ...],
        token_norm: str = "none",
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__(checkpoint, layers, token_norm, device)
        self.pretrained_image_size = int(self.model.config.image_size)
        self.patch_size = int(self.model.config.patch_size)
        parameters = inspect.signature(self.model.forward).parameters
        self.supports_position_interpolation = (
            "interpolate_pos_encoding" in parameters
        )

    @torch.inference_mode()
    def extract(self, pixel_values: torch.Tensor) -> dict[int, torch.Tensor]:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [B, C, H, W]")
        height, width = (int(value) for value in pixel_values.shape[-2:])
        if height != width:
            raise ValueError("High-resolution CLIP inputs must be square")
        clip_patch_grid(height, self.patch_size)
        interpolate = requires_position_interpolation(
            height,
            self.pretrained_image_size,
        )
        if interpolate and not self.supports_position_interpolation:
            raise RuntimeError(
                "Installed transformers CLIPVisionModel does not expose "
                "interpolate_pos_encoding. Use the project requirement "
                "transformers>=4.41,<4.47."
            )

        pixel_values = pixel_values.to(self.device, non_blocking=True)
        kwargs: dict[str, object] = {
            "pixel_values": pixel_values,
            "output_hidden_states": True,
        }
        if self.supports_position_interpolation:
            kwargs["interpolate_pos_encoding"] = interpolate
        outputs = self.model(**kwargs)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("CLIP model did not return hidden states")

        expected_patch_count = (height // self.patch_size) ** 2
        features: dict[int, torch.Tensor] = {}
        for layer in self.layers:
            tokens = hidden_states[layer][:, 1:, :]
            if tokens.shape[1] != expected_patch_count:
                raise RuntimeError(
                    "Interpolated CLIP token count does not match input grid: "
                    f"expected {expected_patch_count}, got {tokens.shape[1]}"
                )
            if self.token_norm == "final_layernorm":
                tokens = self.model.vision_model.post_layernorm(tokens)
            features[layer] = F.normalize(tokens.float(), dim=-1)
        return features
