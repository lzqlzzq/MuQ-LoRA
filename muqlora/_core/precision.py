import torch
from torch import nn

from muqlora.module import LoRAConv1d, LoRALinear, _REDUCED_BASE_DTYPES


NORM_MODULE_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)


def dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def resolve_keep_norm_fp32(
    base_dtype: torch.dtype,
    keep_norm_fp32: bool,
    runtime_device: torch.device | str | None,
) -> bool:
    """Preserve the caller's norm policy on every backend."""
    del base_dtype, runtime_device
    return keep_norm_fp32


def resolve_frontend_dtype(
    base_dtype: torch.dtype,
    runtime_device: torch.device | str | None,
) -> torch.dtype:
    """Keep MuQ's convolutional frontend in FP32 for reduced-precision bases."""
    del runtime_device
    return torch.float32 if base_dtype in _REDUCED_BASE_DTYPES else base_dtype


def apply_precision_policy(wrapper):
    """Place frozen MuQ, norms, adapters, and optional heads in their target dtypes."""
    frontend_tensor_ids = {
        *(id(parameter) for parameter in wrapper.model.model.conv.parameters()),
        *(id(buffer) for buffer in wrapper.model.model.conv.buffers()),
    }

    # Do not use model.to(base_dtype) here. It would round the frontend FP32
    # weights before converting them back, leaving irreversible precision loss.
    for module in wrapper.model.modules():
        for parameter in module.parameters(recurse=False):
            if (
                parameter is not None
                and parameter.is_floating_point()
                and id(parameter) not in frontend_tensor_ids
            ):
                parameter.data = parameter.data.to(dtype=wrapper.base_dtype)
        for name, buffer in module.named_buffers(recurse=False):
            if (
                buffer is not None
                and buffer.is_floating_point()
                and id(buffer) not in frontend_tensor_ids
            ):
                module._buffers[name] = buffer.to(dtype=wrapper.base_dtype)

    if wrapper.keep_norm_fp32:
        for module in wrapper.model.modules():
            if isinstance(module, NORM_MODULE_TYPES):
                module.to(dtype=torch.float32)

    wrapper.model.model.conv.to(dtype=wrapper.frontend_dtype)

    for module in wrapper.model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            module.compute_dtype = wrapper.base_dtype
            module.lora_A.to(dtype=wrapper.adapter_dtype)
            module.lora_B.to(dtype=wrapper.adapter_dtype)

    if wrapper.task_head is not None:
        target_device = next(wrapper.model.parameters()).device
        wrapper.task_head.to(device=target_device, dtype=wrapper.adapter_dtype)
        wrapper.task_head.requires_grad_(True)


def install_norm_precision_hooks(wrapper):
    """Bridge reduced activations through FP32 normalization modules."""
    if not wrapper.keep_norm_fp32:
        return

    frontend_module_ids = {id(module) for module in wrapper.model.model.conv.modules()}
    for module in wrapper.model.modules():
        if id(module) in frontend_module_ids or not isinstance(module, NORM_MODULE_TYPES):
            continue

        input_dtypes = []

        def pre_hook(_module, inputs, input_dtypes=input_dtypes):
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                input_dtypes.append(None)
                return None
            x = inputs[0]
            if x.is_floating_point() and x.dtype != torch.float32:
                input_dtypes.append(x.dtype)
                return (x.float(), *inputs[1:])
            input_dtypes.append(None)
            return None

        def post_hook(_module, _inputs, output, input_dtypes=input_dtypes):
            input_dtype = input_dtypes.pop()
            if input_dtype is None or not isinstance(output, torch.Tensor):
                return output
            return output.to(dtype=input_dtype)

        wrapper._norm_precision_hook_handles.extend(
            (
                module.register_forward_pre_hook(pre_hook),
                module.register_forward_hook(post_hook),
            )
        )


def assert_tensor_dtype(
    tensor: torch.Tensor,
    expected_dtype: torch.dtype,
    description: str,
):
    if tensor.is_floating_point() and tensor.dtype != expected_dtype:
        raise AssertionError(
            f"{description} has dtype {tensor.dtype}, expected {expected_dtype}"
        )


def assert_dtype_policy(wrapper):
    """Raise ``AssertionError`` when the configured precision policy is violated."""
    adapter_parameter_ids = set()
    norm_parameter_ids = set()
    norm_buffer_ids = set()
    frontend_parameter_ids = {
        id(parameter) for parameter in wrapper.model.model.conv.parameters()
    }
    frontend_buffer_ids = {
        id(buffer) for buffer in wrapper.model.model.conv.buffers()
    }

    for module in wrapper.model.modules():
        if isinstance(module, (LoRALinear, LoRAConv1d)):
            if module.compute_dtype != wrapper.base_dtype:
                raise AssertionError(
                    f"{module.__class__.__name__} compute dtype is {module.compute_dtype}, "
                    f"expected {wrapper.base_dtype}"
                )
            for parameter in module.lora_A.parameters():
                adapter_parameter_ids.add(id(parameter))
                if not parameter.requires_grad:
                    raise AssertionError("LoRA A parameter must require gradients")
            for parameter in module.lora_B.parameters():
                adapter_parameter_ids.add(id(parameter))
                if not parameter.requires_grad:
                    raise AssertionError("LoRA B parameter must require gradients")
        if wrapper.keep_norm_fp32 and isinstance(module, NORM_MODULE_TYPES):
            for parameter in module.parameters(recurse=False):
                norm_parameter_ids.add(id(parameter))
            for buffer in module.buffers(recurse=False):
                norm_buffer_ids.add(id(buffer))

    for name, parameter in wrapper.model.named_parameters():
        if id(parameter) in adapter_parameter_ids:
            assert_tensor_dtype(parameter, wrapper.adapter_dtype, f"LoRA parameter {name}")
            continue

        expected_dtype = wrapper.frontend_dtype if id(parameter) in frontend_parameter_ids else (
            torch.float32 if id(parameter) in norm_parameter_ids else wrapper.base_dtype
        )
        assert_tensor_dtype(parameter, expected_dtype, f"frozen base parameter {name}")
        if parameter.requires_grad:
            raise AssertionError(f"frozen base parameter {name} must not require gradients")

    for name, buffer in wrapper.model.named_buffers():
        expected_dtype = wrapper.frontend_dtype if id(buffer) in frontend_buffer_ids else (
            torch.float32 if id(buffer) in norm_buffer_ids else wrapper.base_dtype
        )
        assert_tensor_dtype(buffer, expected_dtype, f"base buffer {name}")

    if wrapper.task_head is not None:
        for name, parameter in wrapper.task_head.named_parameters():
            assert_tensor_dtype(parameter, wrapper.adapter_dtype, f"task-head parameter {name}")
            if not parameter.requires_grad:
                raise AssertionError(f"task-head parameter {name} must require gradients")
