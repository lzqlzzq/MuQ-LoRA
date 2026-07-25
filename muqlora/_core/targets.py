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


def iter_injection_targets(
    model,
    target_modules: tuple[str, ...],
    num_target_layers: int,
):
    layer_count = num_target_layers
    layers = list(model.model.conformer.layers)
    for layer_index in range(len(layers) - 1, -1, -1):
        layer = layers[layer_index]
        if layer_count:
            for name, module in layer.named_modules():
                if name.split(".")[-1] in target_modules:
                    yield (
                        f"model.conformer.layers.{layer_index}.{name}",
                        module,
                    )

            layer_count -= 1


def replace_target_module(model, target_path: str, replacement: nn.Module):
    parent_path, name = target_path.rsplit(".", 1)
    parent = model.get_submodule(parent_path)
    setattr(parent, name, replacement)


def inject_lora_targets(
    model,
    target_modules: tuple[str, ...],
    num_target_layers: int,
    r: int,
    alpha: float,
    compute_dtype: torch.dtype,
) -> list[str]:
    lora_target_names: list[str] = []

    for target_path, module in iter_injection_targets(
        model,
        target_modules=target_modules,
        num_target_layers=num_target_layers,
    ):
        if isinstance(module, nn.Linear):
            replacement = LoRALinear(
                module,
                r,
                alpha,
                compute_dtype=compute_dtype,
            )
        elif isinstance(module, nn.Conv1d):
            replacement = LoRAConv1d(
                module,
                r,
                alpha,
                compute_dtype=compute_dtype,
            )
        else:
            target_name = target_path.split(".", 4)[-1]
            raise TypeError(
                f"target module {target_name!r} is not nn.Linear or nn.Conv1d"
            )

        replace_target_module(model, target_path, replacement)
        lora_target_names.append(target_path)

    return lora_target_names


def iter_lora_modules(model):
    for name, module in model.named_modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            yield name, module


def merge_lora_targets(model) -> list[str]:
    merged_target_names = []

    # Materialize the targets before replacing wrappers in the module tree.
    for target_path, wrapper in list(iter_lora_modules(model)):
        base_module = wrapper.module
        base_weight = base_module.weight

        if isinstance(wrapper, LoRALinear):
            lora_delta = (
                wrapper.lora_B.weight.detach().float()
                @ wrapper.lora_A.weight.detach().float()
            )
        else:
            lora_delta = (
                wrapper.lora_B.weight.detach().float().flatten(1)
                @ wrapper.lora_A.weight.detach().float().flatten(1)
            ).reshape_as(base_weight)

        # Accumulate in FP32 and cast only the final merged value back to the
        # base dtype. This avoids prematurely rounding the LoRA update.
        merged_weight = torch.add(
            base_weight.detach().float(),
            lora_delta,
            alpha=wrapper.scaling,
        )
        with torch.no_grad():
            base_weight.copy_(merged_weight.to(dtype=base_weight.dtype))

        replace_target_module(model, target_path, base_module)
        merged_target_names.append(target_path)

    return merged_target_names


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
