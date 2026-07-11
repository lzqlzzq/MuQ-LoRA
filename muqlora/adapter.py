import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from muqlora._core.targets import validate_adapter_tensor_keys
from muqlora.head import MuQTaskHead


ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"


def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "save_pretrained/load_adapter require the 'safetensors' package"
        ) from exc
    return load_file, save_file


@dataclass
class MuQLoRAConfiguration:
    format_version: int
    base_model_name_or_path: str | None
    target_modules: list[str]
    num_target_layers: int
    precision: dict[str, Any]
    target_manifest: list[dict[str, Any]]
    head_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "base_model_name_or_path": self.base_model_name_or_path,
            "target_modules": self.target_modules,
            "num_target_layers": self.num_target_layers,
            "precision": self.precision,
            "target_manifest": self.target_manifest,
            "head_config": self.head_config,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MuQLoRAConfiguration":
        forbidden = sorted({"adapter_id", "adapter_name", "topology"} & set(data))
        if forbidden:
            raise ValueError(f"adapter_config.json contains unsupported fields: {forbidden}")
        required = {
            "format_version",
            "base_model_name_or_path",
            "target_modules",
            "num_target_layers",
            "precision",
            "target_manifest",
            "head_config",
        }
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"adapter_config.json missing required fields: {missing}")
        if data["format_version"] != 1:
            raise ValueError("unsupported adapter_config.json format_version")
        if not isinstance(data["target_manifest"], list) or not data["target_manifest"]:
            raise ValueError("adapter_config.json must contain a non-empty target_manifest")
        return cls(
            format_version=int(data["format_version"]),
            base_model_name_or_path=data["base_model_name_or_path"],
            target_modules=list(data["target_modules"]),
            num_target_layers=int(data["num_target_layers"]),
            precision=dict(data["precision"]),
            target_manifest=[dict(entry) for entry in data["target_manifest"]],
            head_config=None if data["head_config"] is None else dict(data["head_config"]),
        )


@dataclass
class MuQLoRAAdapter:
    configuration: MuQLoRAConfiguration
    tensors: dict[str, torch.Tensor]
    task_head: MuQTaskHead | None = None

    @classmethod
    def from_model(cls, model) -> "MuQLoRAAdapter":
        return model.current_adapter()

    @classmethod
    def load(
        cls,
        path: str | Path,
        task_head: MuQTaskHead | None = None,
    ) -> "MuQLoRAAdapter":
        load_file, _ = _require_safetensors()
        path = Path(path)
        with (path / ADAPTER_CONFIG_NAME).open("r", encoding="utf-8") as config_file:
            configuration = MuQLoRAConfiguration.from_dict(json.load(config_file))
        if configuration.head_config is not None:
            if task_head is None:
                raise ValueError("adapter package contains task-head weights; pass task_head")
            if not isinstance(task_head, MuQTaskHead):
                raise TypeError("task_head must be a MuQTaskHead")
            if task_head.head_type != configuration.head_config.get("head_type"):
                raise ValueError(
                    "task head type mismatch: "
                    f"saved={configuration.head_config.get('head_type')}, "
                    f"current={task_head.head_type}"
                )
        elif task_head is not None:
            raise ValueError("adapter package does not contain task-head weights")
        tensors = load_file(str(path / ADAPTER_WEIGHTS_NAME))
        validate_adapter_tensor_keys(configuration, tensors, task_head)
        if task_head is not None:
            head_state = {
                name: tensors[f"head.{name}"].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                for name, parameter in task_head.state_dict().items()
            }
            task_head.load_state_dict(head_state)
        return cls(configuration=configuration, tensors=tensors, task_head=task_head)

    def save(self, path: str | Path):
        _, save_file = _require_safetensors()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        validate_adapter_tensor_keys(self.configuration, self.tensors, self.task_head)
        with (path / ADAPTER_CONFIG_NAME).open("w", encoding="utf-8") as config_file:
            json.dump(self.configuration.to_dict(), config_file, indent=2, sort_keys=True)
            config_file.write("\n")
        save_file(
            {key: tensor.detach().cpu().contiguous() for key, tensor in self.tensors.items()},
            str(path / ADAPTER_WEIGHTS_NAME),
        )
