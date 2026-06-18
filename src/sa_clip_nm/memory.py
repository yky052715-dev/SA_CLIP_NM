from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class LayerMemoryBank:
    features: torch.Tensor
    coordinates: torch.Tensor
    source_ids: torch.Tensor

    def validate(self) -> None:
        count = self.features.shape[0]
        if self.coordinates.shape != (count, 2):
            raise ValueError("Memory coordinates must have shape [N, 2]")
        if self.source_ids.shape != (count,):
            raise ValueError("Memory source_ids must have shape [N]")
        if not torch.isfinite(self.features).all():
            raise ValueError("Memory features contain non-finite values")


def _random_indices(count: int, sample_count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randperm(count, generator=generator)[:sample_count]


def _kcenter_indices(
    features: torch.Tensor,
    sample_count: int,
    seed: int,
    projection_dim: int,
    max_candidates: int,
) -> torch.Tensor:
    cpu_features = features.detach().float().cpu()
    count, dimension = cpu_features.shape
    candidate_indices = torch.arange(count)
    if count > max_candidates:
        candidate_indices = _random_indices(count, max_candidates, seed)
        cpu_features = cpu_features[candidate_indices]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if projection_dim > 0 and projection_dim < dimension:
        projection = torch.randn(
            dimension,
            projection_dim,
            generator=generator,
            dtype=cpu_features.dtype,
        ) / projection_dim**0.5
        projected = cpu_features @ projection
    else:
        projected = cpu_features
    projected = F.normalize(projected, dim=-1)

    sample_count = min(sample_count, projected.shape[0])
    first = int(torch.randint(projected.shape[0], (1,), generator=generator).item())
    selected = [first]
    min_distances = 1.0 - projected @ projected[first]
    for _ in range(1, sample_count):
        next_index = int(torch.argmax(min_distances).item())
        selected.append(next_index)
        distances = 1.0 - projected @ projected[next_index]
        min_distances = torch.minimum(min_distances, distances)
    return candidate_indices[torch.tensor(selected, dtype=torch.long)]


def sample_memory_indices(
    features: torch.Tensor,
    ratio: float,
    method: str,
    seed: int,
    projection_dim: int = 128,
    max_candidates: int = 50000,
) -> torch.Tensor:
    if not 0.0 < ratio <= 1.0:
        raise ValueError("Memory ratio must be in (0, 1]")
    count = features.shape[0]
    sample_count = max(1, int(round(count * ratio)))
    if sample_count >= count:
        return torch.arange(count)
    if method == "random":
        return _random_indices(count, sample_count, seed)
    if method == "kcenter":
        return _kcenter_indices(features, sample_count, seed, projection_dim, max_candidates)
    raise ValueError(f"Unsupported memory sampling method: {method}")


def build_layer_memory(
    image_features: torch.Tensor,
    patch_coordinates: torch.Tensor,
    ratio: float,
    method: str,
    seed: int,
    projection_dim: int,
    max_candidates: int,
) -> LayerMemoryBank:
    if image_features.ndim != 3:
        raise ValueError("image_features must have shape [images, patches, dim]")
    image_count, patch_count, dimension = image_features.shape
    if patch_coordinates.shape != (patch_count, 2):
        raise ValueError("patch_coordinates do not match image_features")

    features = image_features.reshape(image_count * patch_count, dimension).cpu()
    coordinates = patch_coordinates.cpu().repeat(image_count, 1)
    source_ids = torch.arange(image_count).repeat_interleave(patch_count)
    indices = sample_memory_indices(
        features,
        ratio=ratio,
        method=method,
        seed=seed,
        projection_dim=projection_dim,
        max_candidates=max_candidates,
    )
    bank = LayerMemoryBank(
        features=features[indices].contiguous(),
        coordinates=coordinates[indices].contiguous(),
        source_ids=source_ids[indices].contiguous(),
    )
    bank.validate()
    return bank


def save_memory_banks(
    banks: dict[int, LayerMemoryBank],
    metadata: dict[str, object],
    path: str | Path,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "banks": {
            int(layer): {
                "features": bank.features.half(),
                "coordinates": bank.coordinates,
                "source_ids": bank.source_ids,
            }
            for layer, bank in banks.items()
        },
    }
    torch.save(payload, output_path)


def load_memory_banks(path: str | Path) -> tuple[dict[int, LayerMemoryBank], dict[str, object]]:
    payload = torch.load(Path(path), map_location="cpu")
    banks = {
        int(layer): LayerMemoryBank(
            features=data["features"].float(),
            coordinates=data["coordinates"].float(),
            source_ids=data["source_ids"].long(),
        )
        for layer, data in payload["banks"].items()
    }
    for bank in banks.values():
        bank.validate()
    return banks, payload["metadata"]

