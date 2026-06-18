from __future__ import annotations

import importlib
import json
import platform

import torch


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return str(getattr(module, "__version__", "unknown"))
    except ImportError:
        return "not installed"


def main() -> None:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": module_version("transformers"),
        "numpy": module_version("numpy"),
        "sklearn": module_version("sklearn"),
        "scipy": module_version("scipy"),
        "PIL": module_version("PIL"),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

