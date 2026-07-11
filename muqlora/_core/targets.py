from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from muqlora.head import MuQTaskHead
from muqlora.module import LoRAConv1d, LoRALinear


def target_tensor_keys(target_path: str) -> tuple[str, str]:
    return (
        f"lora.{target_path}.lora_A.weight",
        f"lora.{target_path}.lora_B.weight",
    )


def validate_adapter_tensor_keys(
    configuration,
    tensors: Mapping[str, torch.Tensor],
    task_head: MuQTaskHead | None,
):
    expected = set()
    for entry in configuration.target_manifest:
        expected.update(target_tensor_keys(entry["target_path"]))
    if configuration.head_config is not None:
        if task_head is None:
            raise ValueError("task_head is required to validate saved task-head tensors")
        expected.update(f"head.{name}" for name in task_head.state_dict())
    actual = set(tensors)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"adapter tensor keys mismatch: missing={missing}, extra={extra}")


def inject_lora_targets(
    model,
    target_modules: tuple[str, ...],
    num_target_layers: int,
    r: int,
    alpha: float,
    compute_dtype: torch.dtype,
) -> list[str]:
    lora_target_names: list[str] = []
    layer_count = num_target_layers

    layers = list(model.model.conformer.layers)
    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        if layer_count:
            for name, module in layer.named_modules():
                if name.split(".")[-1] in target_modules:
                    if isinstance(module, nn.Linear):
                        module = LoRALinear(
                            module,
                            r,
                            alpha,
                            compute_dtype=compute_dtype,
                        )
                    elif isinstance(module, nn.Conv1d):
                        module = LoRAConv1d(
                            module,
                            r,
                            alpha,
                            compute_dtype=compute_dtype,
                        )
                    else:
                        raise TypeError(
                            f"target module {name!r} is not nn.Linear or nn.Conv1d"
                        )

                    parent = layer
                    *path, last = name.split(".")
                    for p in path:
                        parent = getattr(parent, p)
                    setattr(parent, last, module)
                    lora_target_names.append(
                        f"model.conformer.layers.{layer_index}.{name}"
                    )

            layer_count -= 1

    return lora_target_names


def iter_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            yield name, module


def target_manifest(model) -> list[dict[str, Any]]:
    return [
        module.target_manifest_entry(target_path)
        for target_path, module in iter_lora_modules(model)
    ]


def adapter_weight_tensors(
    model,
    task_head: MuQTaskHead | None,
) -> dict[str, torch.Tensor]:
    tensors = {}
    for target_path, module in iter_lora_modules(model):
        lora_a_key, lora_b_key = target_tensor_keys(target_path)
        tensors[lora_a_key] = module.lora_A.weight.detach().cpu().contiguous()
        tensors[lora_b_key] = module.lora_B.weight.detach().cpu().contiguous()

    if task_head is not None:
        for name, tensor in task_head.state_dict().items():
            tensors[f"head.{name}"] = tensor.detach().cpu().contiguous()
    return tensors


def validate_adapter_manifest(
    adapter,
    current_manifest: list[dict[str, Any]],
    target_modules: tuple[str, ...],
    num_target_layers: int,
):
    if adapter.configuration.target_manifest != current_manifest:
        raise ValueError(
            "adapter target_manifest mismatch: "
            f"saved={adapter.configuration.target_manifest}, current={current_manifest}"
        )
    if adapter.configuration.target_modules != list(target_modules):
        raise ValueError(
            "adapter target_modules mismatch: "
            f"saved={adapter.configuration.target_modules}, current={list(target_modules)}"
        )
    if adapter.configuration.num_target_layers != num_target_layers:
        raise ValueError(
            "adapter num_target_layers mismatch: "
            f"saved={adapter.configuration.num_target_layers}, current={num_target_layers}"
        )
    if adapter.configuration.head_config is not None:
        if adapter.task_head is None:
            raise ValueError("adapter contains task-head weights but has no task_head")
        if adapter.task_head.head_type != adapter.configuration.head_config.get("head_type"):
            raise ValueError(
                "task head type mismatch: "
                f"saved={adapter.configuration.head_config.get('head_type')}, "
                f"current={adapter.task_head.head_type}"
            )
    elif adapter.task_head is not None:
        raise ValueError("adapter configuration does not contain task-head metadata")
    validate_adapter_tensor_keys(
        adapter.configuration,
        adapter.tensors,
        adapter.task_head,
    )


def copy_lora_tensors(model, tensors: Mapping[str, torch.Tensor]):
    for target_path, module in iter_lora_modules(model):
        lora_a_key, lora_b_key = target_tensor_keys(target_path)
        module.lora_A.weight.data.copy_(
            tensors[lora_a_key].to(
                device=module.lora_A.weight.device,
                dtype=module.lora_A.weight.dtype,
            )
        )
        module.lora_B.weight.data.copy_(
            tensors[lora_b_key].to(
                device=module.lora_B.weight.device,
                dtype=module.lora_B.weight.dtype,
            )
        )
