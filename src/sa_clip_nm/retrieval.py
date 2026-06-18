from __future__ import annotations

import torch
import torch.nn.functional as F

from .memory import LayerMemoryBank


@torch.inference_mode()
def exact_spatial_nearest_neighbor(
    query_features: torch.Tensor,
    query_coordinates: torch.Tensor,
    bank: LayerMemoryBank,
    spatial_weight: float,
    query_chunk_size: int = 256,
    bank_chunk_size: int = 4096,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return minimum joint distances and memory indices for each query patch."""
    if query_features.ndim != 2:
        raise ValueError("query_features must have shape [Q, D]")
    if query_coordinates.shape != (query_features.shape[0], 2):
        raise ValueError("query_coordinates must have shape [Q, 2]")
    if spatial_weight < 0:
        raise ValueError("spatial_weight must be non-negative")

    target_device = torch.device(device)
    query_features = F.normalize(query_features.float(), dim=-1).to(target_device)
    query_coordinates = query_coordinates.float().to(target_device)
    bank_features_cpu = F.normalize(bank.features.float(), dim=-1)
    bank_coordinates_cpu = bank.coordinates.float()

    all_scores: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for query_start in range(0, query_features.shape[0], query_chunk_size):
        query_end = min(query_start + query_chunk_size, query_features.shape[0])
        query_chunk = query_features[query_start:query_end]
        query_coordinate_chunk = query_coordinates[query_start:query_end]
        chunk_best_scores = torch.full(
            (query_chunk.shape[0],),
            float("inf"),
            dtype=torch.float32,
            device=target_device,
        )
        chunk_best_indices = torch.full(
            (query_chunk.shape[0],),
            -1,
            dtype=torch.long,
            device=target_device,
        )

        for bank_start in range(0, bank.features.shape[0], bank_chunk_size):
            bank_end = min(bank_start + bank_chunk_size, bank.features.shape[0])
            bank_features = bank_features_cpu[bank_start:bank_end].to(
                target_device, non_blocking=True
            )
            feature_distance = 1.0 - query_chunk @ bank_features.T
            if spatial_weight > 0:
                bank_coordinates = bank_coordinates_cpu[bank_start:bank_end].to(
                    target_device, non_blocking=True
                )
                position_distance = torch.cdist(
                    query_coordinate_chunk,
                    bank_coordinates,
                    p=2,
                )
                joint_distance = feature_distance + spatial_weight * position_distance
            else:
                joint_distance = feature_distance

            local_scores, local_indices = joint_distance.min(dim=1)
            update = local_scores < chunk_best_scores
            chunk_best_scores[update] = local_scores[update]
            chunk_best_indices[update] = local_indices[update] + bank_start

        all_scores.append(chunk_best_scores.cpu())
        all_indices.append(chunk_best_indices.cpu())

    return torch.cat(all_scores), torch.cat(all_indices)


def batched_layer_scores(
    image_features: torch.Tensor,
    patch_coordinates: torch.Tensor,
    bank: LayerMemoryBank,
    spatial_weight: float,
    query_chunk_size: int,
    bank_chunk_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    if image_features.ndim != 3:
        raise ValueError("image_features must have shape [B, P, D]")
    image_scores = []
    for features in image_features:
        scores, _ = exact_spatial_nearest_neighbor(
            query_features=features,
            query_coordinates=patch_coordinates,
            bank=bank,
            spatial_weight=spatial_weight,
            query_chunk_size=query_chunk_size,
            bank_chunk_size=bank_chunk_size,
            device=device,
        )
        image_scores.append(scores)
    return torch.stack(image_scores)

