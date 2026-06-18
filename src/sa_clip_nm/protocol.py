from __future__ import annotations

import hashlib
import json
from typing import Any


def threshold_parameters(config: dict[str, Any]) -> dict[str, Any]:
    calibration = config["calibration"]
    return {
        "pixel_threshold_method": str(
            calibration["pixel_threshold_method"]
        ),
        "pixel_quantile": float(calibration["pixel_quantile"]),
        "pixel_image_quantile": float(
            calibration["pixel_image_quantile"]
        ),
        "pixel_topk_fraction": float(
            calibration["pixel_topk_fraction"]
        ),
    }


def metric_protocol(
    config: dict[str, Any],
    include_threshold_method: bool,
) -> dict[str, Any]:
    protocol = {
        "model_seed": int(config["experiment"]["seed"]),
        "data": {
            "dataset": str(config["data"]["dataset"]),
            "image_size": int(config["data"]["image_size"]),
            "resize_mode": str(config["data"]["resize_mode"]),
            "calibration_fraction": float(
                config["data"]["calibration_fraction"]
            ),
        },
        "model": {
            "checkpoint": str(config["model"]["checkpoint"]),
            "active_layers": [
                int(value) for value in config["model"]["active_layers"]
            ],
            "token_norm": str(config["model"]["token_norm"]),
        },
        "memory": {
            "sampling": str(config["memory"]["sampling"]),
            "ratio": float(config["memory"]["ratio"]),
            "projection_dim": int(config["memory"]["projection_dim"]),
            "max_candidates": int(config["memory"]["max_candidates"]),
        },
        "retrieval": {
            "spatial_mode": str(config["retrieval"]["spatial_mode"]),
            "fixed_lambda": float(config["retrieval"]["fixed_lambda"]),
            "query_chunk_size": int(
                config["retrieval"]["query_chunk_size"]
            ),
            "bank_chunk_size": int(
                config["retrieval"]["bank_chunk_size"]
            ),
            "lambda_max": float(config["retrieval"]["lambda_max"]),
            "tau_quantile": float(config["retrieval"]["tau_quantile"]),
        },
        "calibration": {
            "mad_epsilon": float(config["calibration"]["mad_epsilon"]),
            "threshold_fit_fraction": float(
                config["calibration"]["threshold_fit_fraction"]
            ),
            "pixel_quantile": float(
                config["calibration"]["pixel_quantile"]
            ),
            "pixel_image_quantile": float(
                config["calibration"]["pixel_image_quantile"]
            ),
            "pixel_topk_fraction": float(
                config["calibration"]["pixel_topk_fraction"]
            ),
            "image_quantile": float(
                config["calibration"]["image_quantile"]
            ),
            "clamp_min_zero": bool(
                config["calibration"]["clamp_min_zero"]
            ),
        },
        "inference": {
            "gaussian_sigma": float(
                config["inference"]["gaussian_sigma"]
            ),
            "image_score": str(config["inference"]["image_score"]),
            "image_topk_fraction": float(
                config["inference"]["image_topk_fraction"]
            ),
        },
        "evaluation": {
            "compute_oracle_f1": bool(
                config["evaluation"]["compute_oracle_f1"]
            ),
        },
    }
    if include_threshold_method:
        protocol["calibration"]["pixel_threshold_method"] = str(
            config["calibration"]["pixel_threshold_method"]
        )
    return protocol


def payload_fingerprint(payload: dict[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def metric_protocol_fingerprint(
    config: dict[str, Any],
    include_threshold_method: bool,
) -> str:
    return payload_fingerprint(
        metric_protocol(config, include_threshold_method)
    )
