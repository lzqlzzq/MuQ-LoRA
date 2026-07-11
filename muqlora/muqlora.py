import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

import muq
from muqlora.module import (
    LoRAConv1d,
    LoRALinear,
    YaRNRotaryPositionalEmbedding,
    _REDUCED_BASE_DTYPES,
    _autocast_for,
)


MUQ_MEL_INPUT_CONFIG = {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 240,
    "n_mels": 128,
    "is_db": True,
}

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"

_NORM_MODULE_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)


def _dtype_to_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _class_path(module: nn.Module) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__qualname__}"


def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "save_pretrained/load_adapter require the 'safetensors' package"
        ) from exc
    return load_file, save_file


class MuQTaskHead(nn.Module, ABC):
    """Base class for MuQLoRA task heads."""

    def __init__(self):
        super().__init__()

    @property
    def head_type(self) -> str:
        return _class_path(self)

    def get_config(self) -> dict[str, Any]:
        return {
            "head_type": self.head_type,
        }

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Mapping[str, torch.Tensor]:
        """Return a TensorDict for this task."""


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
    def from_model(cls, model: "MuQLoRA") -> "MuQLoRAAdapter":
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
        _validate_adapter_tensor_keys(configuration, tensors, task_head)
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
        _validate_adapter_tensor_keys(self.configuration, self.tensors, self.task_head)
        with (path / ADAPTER_CONFIG_NAME).open("w", encoding="utf-8") as config_file:
            json.dump(self.configuration.to_dict(), config_file, indent=2, sort_keys=True)
            config_file.write("\n")
        save_file(
            {key: tensor.detach().cpu().contiguous() for key, tensor in self.tensors.items()},
            str(path / ADAPTER_WEIGHTS_NAME),
        )


def _target_tensor_keys(target_path: str) -> tuple[str, str]:
    return (
        f"lora.{target_path}.lora_A.weight",
        f"lora.{target_path}.lora_B.weight",
    )


def _validate_adapter_tensor_keys(
    configuration: MuQLoRAConfiguration,
    tensors: Mapping[str, torch.Tensor],
    task_head: MuQTaskHead | None,
):
    expected = set()
    for entry in configuration.target_manifest:
        expected.update(_target_tensor_keys(entry["target_path"]))
    if configuration.head_config is not None:
        if task_head is None:
            raise ValueError("task_head is required to validate saved task-head tensors")
        expected.update(f"head.{name}" for name in task_head.state_dict())
    actual = set(tensors)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"adapter tensor keys mismatch: missing={missing}, extra={extra}")


class MuQLoRA(nn.Module):
    """LoRA wrapper for MuQ's Conformer encoder.

    The default path is feature-only: it runs MuQ preprocessing, conv
    subsampling, and the Conformer encoder, while skipping MuQ's original
    codebook projection head. Passing ``task_head`` turns the module into a
    task model that feeds pooled encoder features to that ``MuQTaskHead``.

    Input waveform shape:
        ``[batch_size, timestep]`` or ``[batch_size, audio_channel=1, timestep]``.

    Input mel shape:
        ``[batch_size, n_mels=128, mel_frame_count]`` or
        ``[batch_size, audio_channel=1, n_mels=128, mel_frame_count]`` with
        ``input_type="mel"``. The mel must be generated with MuQ's preprocessing
        parameters: ``sample_rate=24000``, ``n_fft=2048``, ``hop_length=240``,
        ``n_mels=128``, ``is_db=True``. MuQ removes the final mel frame in
        preprocessing, so ``input_type="mel"`` applies the same ``[..., :-1]``
        trim internally. Use ``input_type="muq_mel"`` if the tensor is already
        the trimmed MuQ preprocessing output.

    Feature-only output:
        A BaseModelOutput-like object from the Conformer encoder where
        ``last_hidden_state`` has shape ``[batch_size, frame_count, hidden_size]``.

    Task-head output:
        The active task head's TensorDict. With ``pooling="mean"`` or
        ``"cls"``, the head receives ``[batch_size, hidden_size]``.
        With ``pooling="none"``, the head receives
        ``[batch_size, frame_count, hidden_size]``.

    Precision policy:
        Frozen MuQ weights use ``base_dtype``. With ``keep_norm_fp32=True``,
        LayerNorm, BatchNorm, and GroupNorm parameters and buffers remain
        FP32. LoRA A/B and optional task-head parameters use
        ``adapter_dtype`` storage, while their matmuls execute under local
        autocast at ``base_dtype``.

        When ``base_dtype`` is reduced precision, MuQLoRA runs the complete
        convolutional frontend in FP32, then casts its output once before the
        Conformer encoder. This keeps the frontend BatchNorm running statistics
        in FP32, which avoids amplifying reduced-precision convolution
        rounding. FP32 normalization receives an FP32 activation through an
        internal input/output bridge whenever its caller uses FP16, on every
        backend.
    """

    def __init__(
        self,
        model: muq.MuQ,
        r: int = 8,
        alpha: float = 16.0,
        target_modules: Sequence[str] | None = None,
        num_target_layers: int = 2,
        train_muq_head: bool = False,
        keep_base_model_eval: bool = True,
        feature_only: bool | None = None,
        task_head: MuQTaskHead | None = None,
        pooling: str | None = "mean",
        drop_muq_head: bool | None = None,
        base_dtype: torch.dtype = torch.float16,
        adapter_dtype: torch.dtype = torch.float32,
        base_model_name_or_path: str | None = None,
        keep_norm_fp32: bool = True,
        runtime_device: torch.device | str | None = None,
        yarn_factor: float | None = None,
        yarn_original_max_position_embeddings: int | None = None,
        yarn_attention_factor: float | None = None,
        yarn_beta_fast: float = 32.0,
        yarn_beta_slow: float = 1.0,
        yarn_mscale: float | None = None,
        yarn_mscale_all_dim: float | None = None,
        yarn_truncate: bool = True,
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("r must be positive")
        if num_target_layers < 0:
            raise ValueError("num_target_layers must be non-negative")
        if train_muq_head:
            raise ValueError(
                "train_muq_head is not supported; use feature_only=True with task heads instead"
            )
        if task_head is not None and not isinstance(task_head, MuQTaskHead):
            raise TypeError("task_head must be a MuQTaskHead")
        if task_head is not None and feature_only is False:
            raise ValueError("task_head requires feature_only=True")
        if base_dtype not in (torch.float32, torch.float16):
            raise ValueError("base_dtype must be torch.float32 or torch.float16; BF16 is unsupported")
        if adapter_dtype != torch.float32:
            raise ValueError("adapter_dtype must be torch.float32")

        self.model = model
        self.model.requires_grad_(False)  # Freeze the original model parameters before injecting LoRA.

        self.r = r
        self.alpha = alpha
        self.target_modules = tuple(target_modules or ())
        self.num_target_layers = num_target_layers
        self.keep_base_model_eval = keep_base_model_eval
        self.base_dtype = base_dtype
        self.adapter_dtype = adapter_dtype
        self.base_model_name_or_path = base_model_name_or_path
        self.runtime_device = None if runtime_device is None else torch.device(runtime_device)
        self.keep_norm_fp32 = self.resolve_keep_norm_fp32(
            base_dtype=base_dtype,
            keep_norm_fp32=keep_norm_fp32,
            runtime_device=self.runtime_device,
        )
        self.frontend_dtype = self.resolve_frontend_dtype(
            base_dtype=base_dtype,
            runtime_device=self.runtime_device,
        )
        self._norm_precision_hook_handles = []
        self.task_head = task_head
        self.pooling = pooling
        self.feature_only = self.task_head is not None or (True if feature_only is None else feature_only)
        self.drop_muq_head = self.feature_only if drop_muq_head is None else drop_muq_head
        if self.drop_muq_head and not self.feature_only:
            raise ValueError("drop_muq_head requires feature_only=True")

        if self.drop_muq_head:
            self.model.model.linear = nn.Identity()

        if yarn_factor is not None:
            self._apply_yarn_rotary_embedding(
                factor=yarn_factor,
                original_max_position_embeddings=yarn_original_max_position_embeddings,
                attention_factor=yarn_attention_factor,
                beta_fast=yarn_beta_fast,
                beta_slow=yarn_beta_slow,
                mscale=yarn_mscale,
                mscale_all_dim=yarn_mscale_all_dim,
                truncate=yarn_truncate,
            )

        self._lora_target_names: list[str] = []
        layer_count = num_target_layers

        # Create low-rank matrices for each linear layer in the model
        layers = list(self.model.model.conformer.layers)
        for layer_index in range(len(layers) - 1, -1, -1):
            layer = layers[layer_index]
            if layer_count:
                for name, module in layer.named_modules():
                    if name.split(".")[-1] in self.target_modules:
                        if isinstance(module, nn.Linear):
                            module = LoRALinear(
                                module,
                                r,
                                alpha,
                                compute_dtype=base_dtype,
                            )
                        elif isinstance(module, nn.Conv1d):
                            module = LoRAConv1d(
                                module,
                                r,
                                alpha,
                                compute_dtype=base_dtype,
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
                        self._lora_target_names.append(
                            f"model.conformer.layers.{layer_index}.{name}"
                        )

                layer_count -= 1

        self._apply_precision_policy()
        self.assert_dtype_policy()
        self._install_norm_precision_hooks()
        self.train()

    def _iter_lora_modules(self):
        for name, module in self.model.named_modules():
            if isinstance(module, (LoRALinear, LoRAConv1d)):
                yield name, module

    def _target_manifest(self) -> list[dict[str, Any]]:
        return [
            module.target_manifest_entry(target_path)
            for target_path, module in self._iter_lora_modules()
        ]

    def _adapter_configuration(self) -> MuQLoRAConfiguration:
        return MuQLoRAConfiguration(
            format_version=1,
            base_model_name_or_path=self.base_model_name_or_path,
            target_modules=list(self.target_modules),
            num_target_layers=self.num_target_layers,
            precision={
                "base_dtype": _dtype_to_name(self.base_dtype),
                "adapter_dtype": _dtype_to_name(self.adapter_dtype),
                "frontend_dtype": _dtype_to_name(self.frontend_dtype),
                "keep_norm_fp32": self.keep_norm_fp32,
            },
            target_manifest=self._target_manifest(),
            head_config=None if self.task_head is None else self.task_head.get_config(),
        )

    def _adapter_weight_tensors(self) -> dict[str, torch.Tensor]:
        tensors = {}
        for target_path, module in self._iter_lora_modules():
            lora_a_key, lora_b_key = _target_tensor_keys(target_path)
            tensors[lora_a_key] = module.lora_A.weight.detach().cpu().contiguous()
            tensors[lora_b_key] = module.lora_B.weight.detach().cpu().contiguous()

        if self.task_head is not None:
            for name, tensor in self.task_head.state_dict().items():
                tensors[f"head.{name}"] = tensor.detach().cpu().contiguous()
        return tensors

    def current_adapter(self) -> MuQLoRAAdapter:
        return MuQLoRAAdapter(
            configuration=self._adapter_configuration(),
            tensors=self._adapter_weight_tensors(),
            task_head=self.task_head,
        )

    export_adapter = current_adapter

    def _validate_adapter_manifest(self, adapter: MuQLoRAAdapter):
        if not isinstance(adapter, MuQLoRAAdapter):
            raise TypeError("set_adapter expects a MuQLoRAAdapter")
        current_manifest = self._target_manifest()
        if adapter.configuration.target_manifest != current_manifest:
            raise ValueError(
                "adapter target_manifest mismatch: "
                f"saved={adapter.configuration.target_manifest}, current={current_manifest}"
            )
        if adapter.configuration.target_modules != list(self.target_modules):
            raise ValueError(
                "adapter target_modules mismatch: "
                f"saved={adapter.configuration.target_modules}, current={list(self.target_modules)}"
            )
        if adapter.configuration.num_target_layers != self.num_target_layers:
            raise ValueError(
                "adapter num_target_layers mismatch: "
                f"saved={adapter.configuration.num_target_layers}, current={self.num_target_layers}"
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
        _validate_adapter_tensor_keys(
            adapter.configuration,
            adapter.tensors,
            adapter.task_head,
        )

    def set_adapter(self, adapter: MuQLoRAAdapter):
        self._validate_adapter_manifest(adapter)
        for target_path, module in self._iter_lora_modules():
            lora_a_key, lora_b_key = _target_tensor_keys(target_path)
            module.lora_A.weight.data.copy_(
                adapter.tensors[lora_a_key].to(
                    device=module.lora_A.weight.device,
                    dtype=module.lora_A.weight.dtype,
                )
            )
            module.lora_B.weight.data.copy_(
                adapter.tensors[lora_b_key].to(
                    device=module.lora_B.weight.device,
                    dtype=module.lora_B.weight.dtype,
                )
            )

        self.task_head = adapter.task_head
        if self.task_head is not None:
            target_device = next(self.model.parameters()).device
            self.task_head.to(device=target_device, dtype=self.adapter_dtype)
            head_state = {
                name: adapter.tensors[f"head.{name}"].to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                for name, parameter in self.task_head.state_dict().items()
            }
            self.task_head.load_state_dict(head_state)
            self.task_head.requires_grad_(True)
        self.assert_dtype_policy()

    def save_pretrained(self, path: str | Path):
        warnings.warn(
            "MuQLoRA.save_pretrained() is deprecated; use "
            "MuQLoRAAdapter.from_model(model).save(path) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.current_adapter().save(path)

    def load_adapter(
        self,
        path: str | Path,
        task_head: MuQTaskHead | None = None,
    ) -> MuQLoRAAdapter:
        warnings.warn(
            "MuQLoRA.load_adapter() is deprecated; use "
            "MuQLoRAAdapter.load(path, task_head=...) and model.set_adapter(adapter) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        adapter = MuQLoRAAdapter.load(path, task_head=task_head)
        self.set_adapter(adapter)
        return adapter

    def _apply_yarn_rotary_embedding(
        self,
        factor: float,
        original_max_position_embeddings: int | None,
        attention_factor: float | None,
        beta_fast: float,
        beta_slow: float,
        mscale: float | None,
        mscale_all_dim: float | None,
        truncate: bool,
    ):
        conformer = self.model.model.conformer
        position_embeddings_type = getattr(conformer.config, "position_embeddings_type", None)
        if position_embeddings_type != "rotary":
            raise ValueError(
                "YaRN adapts rotary positional embeddings; "
                f"MuQ conformer position_embeddings_type is {position_embeddings_type!r}"
            )

        conformer.embed_positions = YaRNRotaryPositionalEmbedding(
            conformer.config,
            factor=factor,
            original_max_position_embeddings=original_max_position_embeddings,
            attention_factor=attention_factor,
            beta_fast=beta_fast,
            beta_slow=beta_slow,
            mscale=mscale,
            mscale_all_dim=mscale_all_dim,
            truncate=truncate,
        )

    @staticmethod
    def resolve_keep_norm_fp32(
        base_dtype: torch.dtype,
        keep_norm_fp32: bool,
        runtime_device: torch.device | str | None,
    ) -> bool:
        """Preserve the caller's norm policy on every backend.

        MPS compatibility is implemented at norm call sites, rather than by
        changing the meaning of ``keep_norm_fp32``.
        """
        del base_dtype, runtime_device
        return keep_norm_fp32

    @staticmethod
    def resolve_frontend_dtype(
        base_dtype: torch.dtype,
        runtime_device: torch.device | str | None,
    ) -> torch.dtype:
        """Keep MuQ's convolutional frontend in FP32 for reduced-precision bases."""
        del runtime_device  # Frontend fidelity is backend-independent.
        return torch.float32 if base_dtype in _REDUCED_BASE_DTYPES else base_dtype

    def _apply_precision_policy(self):
        """Place frozen MuQ, norms, adapters, and optional heads in their target dtypes."""
        frontend_tensor_ids = {
            *(id(parameter) for parameter in self.model.model.conv.parameters()),
            *(id(buffer) for buffer in self.model.model.conv.buffers()),
        }

        # Do not use self.model.to(base_dtype) here. It would round the
        # frontend FP32 weights to the base dtype before converting them back,
        # leaving an irreversible FP32 -> reduced -> FP32 weight error.
        for module in self.model.modules():
            for parameter in module.parameters(recurse=False):
                if (
                    parameter is not None
                    and parameter.is_floating_point()
                    and id(parameter) not in frontend_tensor_ids
                ):
                    parameter.data = parameter.data.to(dtype=self.base_dtype)
            for name, buffer in module.named_buffers(recurse=False):
                if (
                    buffer is not None
                    and buffer.is_floating_point()
                    and id(buffer) not in frontend_tensor_ids
                ):
                    module._buffers[name] = buffer.to(dtype=self.base_dtype)

        if self.keep_norm_fp32:
            for module in self.model.modules():
                if isinstance(module, _NORM_MODULE_TYPES):
                    module.to(dtype=torch.float32)

        # The Conv2dSubsampling frontend contains BatchNorm2d layers with very
        # small running variances. Reduced-precision convolution rounding is
        # amplified there, so the complete frontend stays FP32 on every backend.
        self.model.model.conv.to(dtype=self.frontend_dtype)

        for module in self.model.modules():
            if isinstance(module, (LoRALinear, LoRAConv1d)):
                module.compute_dtype = self.base_dtype
                module.lora_A.to(dtype=self.adapter_dtype)
                module.lora_B.to(dtype=self.adapter_dtype)

        if self.task_head is not None:
            target_device = next(self.model.parameters()).device
            self.task_head.to(device=target_device, dtype=self.adapter_dtype)
            self.task_head.requires_grad_(True)

    def _install_norm_precision_hooks(self):
        """Bridge reduced activations through FP32 normalization modules.

        Normalization kernels require activation and FP32 norm-state dtypes to
        agree. The hooks preserve the backend-neutral ``keep_norm_fp32``
        contract by upcasting every reduced-precision norm input and restoring
        its original dtype on output.
        """
        if not self.keep_norm_fp32:
            return

        frontend_module_ids = {id(module) for module in self.model.model.conv.modules()}
        for module in self.model.modules():
            if id(module) in frontend_module_ids or not isinstance(module, _NORM_MODULE_TYPES):
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

            self._norm_precision_hook_handles.extend(
                (
                    module.register_forward_pre_hook(pre_hook),
                    module.register_forward_hook(post_hook),
                )
            )

    @staticmethod
    def _assert_tensor_dtype(
        tensor: torch.Tensor,
        expected_dtype: torch.dtype,
        description: str,
    ):
        if tensor.is_floating_point() and tensor.dtype != expected_dtype:
            raise AssertionError(
                f"{description} has dtype {tensor.dtype}, expected {expected_dtype}"
            )

    def assert_dtype_policy(self):
        """Raise ``AssertionError`` when the configured precision policy is violated."""
        adapter_parameter_ids = set()
        norm_parameter_ids = set()
        norm_buffer_ids = set()
        frontend_parameter_ids = {
            id(parameter) for parameter in self.model.model.conv.parameters()
        }
        frontend_buffer_ids = {
            id(buffer) for buffer in self.model.model.conv.buffers()
        }

        for module in self.model.modules():
            if isinstance(module, (LoRALinear, LoRAConv1d)):
                if module.compute_dtype != self.base_dtype:
                    raise AssertionError(
                        f"{module.__class__.__name__} compute dtype is {module.compute_dtype}, "
                        f"expected {self.base_dtype}"
                    )
                for parameter in module.lora_A.parameters():
                    adapter_parameter_ids.add(id(parameter))
                    if not parameter.requires_grad:
                        raise AssertionError("LoRA A parameter must require gradients")
                for parameter in module.lora_B.parameters():
                    adapter_parameter_ids.add(id(parameter))
                    if not parameter.requires_grad:
                        raise AssertionError("LoRA B parameter must require gradients")
            if self.keep_norm_fp32 and isinstance(module, _NORM_MODULE_TYPES):
                for parameter in module.parameters(recurse=False):
                    norm_parameter_ids.add(id(parameter))
                for buffer in module.buffers(recurse=False):
                    norm_buffer_ids.add(id(buffer))

        for name, parameter in self.model.named_parameters():
            if id(parameter) in adapter_parameter_ids:
                self._assert_tensor_dtype(parameter, self.adapter_dtype, f"LoRA parameter {name}")
                continue

            expected_dtype = self.frontend_dtype if id(parameter) in frontend_parameter_ids else (
                torch.float32 if id(parameter) in norm_parameter_ids else self.base_dtype
            )
            self._assert_tensor_dtype(parameter, expected_dtype, f"frozen base parameter {name}")
            if parameter.requires_grad:
                raise AssertionError(f"frozen base parameter {name} must not require gradients")

        for name, buffer in self.model.named_buffers():
            expected_dtype = self.frontend_dtype if id(buffer) in frontend_buffer_ids else (
                torch.float32 if id(buffer) in norm_buffer_ids else self.base_dtype
            )
            self._assert_tensor_dtype(buffer, expected_dtype, f"base buffer {name}")

        if self.task_head is not None:
            for name, parameter in self.task_head.named_parameters():
                self._assert_tensor_dtype(parameter, self.adapter_dtype, f"task-head parameter {name}")
                if not parameter.requires_grad:
                    raise AssertionError(f"task-head parameter {name} must require gradients")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.keep_base_model_eval:
            self.model.eval()
            for module in self.modules():
                if isinstance(module, (LoRALinear, LoRAConv1d)):
                    module.train(mode)
        return self

    def prepare_encoder_inputs(
        self,
        x,
        attention_mask: torch.Tensor | None = None,
        input_type: str = "waveform",
    ):
        """Convert waveform or mel features to Conformer-ready frames.

        Args:
            x: Raw waveform shaped ``[batch_size, timestep]`` or
                ``[batch_size, 1, timestep]`` when ``input_type="waveform"``.
                Raw dB mel shaped ``[batch_size, 128, mel_frame_count]`` or
                ``[batch_size, 1, 128, mel_frame_count]`` when
                ``input_type="mel"``. Already-trimmed MuQ mel features with the
                same mel shape when ``input_type="muq_mel"``.
            attention_mask: Optional mask. For waveform input this is shaped
                ``[batch_size, timestep]``. For mel input this is shaped
                ``[batch_size, mel_frame_count]``.
            input_type: ``"waveform"``, ``"mel"``, or ``"muq_mel"``.

        Returns:
            A pair ``(hidden_states, encoder_attention_mask)`` where
            ``hidden_states`` is shaped ``[batch_size, frame_count, hidden_size]``
            after MuQ's normalization and conv subsampling, and
            ``encoder_attention_mask`` is downsampled to
            ``[batch_size, frame_count]`` when provided.
        """
        muq_model = self.model.model

        if input_type == "waveform":
            features = muq_model.preprocessing(x, features=["melspec_2048"])
        elif input_type == "mel":
            features = {"melspec_2048": x[..., :-1]}
        elif input_type == "muq_mel":
            features = {"melspec_2048": x}
        else:
            raise ValueError(f"unsupported input_type: {input_type!r}")

        features = muq_model.normalize(features)
        mel_features = features["melspec_2048"].to(dtype=self.frontend_dtype)
        hidden_states = muq_model.conv(mel_features).to(dtype=self.base_dtype)

        if attention_mask is not None:
            if input_type == "mel":
                attention_mask = attention_mask[..., :-1]
            attention_mask = attention_mask.bool()
            skip_n = int(attention_mask.size(-1) / hidden_states.size(1))
            if skip_n <= 0:
                raise ValueError("attention_mask is shorter than the encoded sequence")
            attention_mask = attention_mask[:, ::skip_n]
            attention_mask = attention_mask[:, : hidden_states.size(1)]

        return hidden_states, attention_mask

    def encode(
        self,
        x,
        attention_mask: torch.Tensor | None = None,
        input_type: str = "waveform",
        output_attentions: bool = False,
        output_hidden_states: bool = True,
        return_dict: bool = True,
        return_attention_mask: bool = False,
    ):
        """Run the MuQ encoder without computing the original codebook logits.

        Args:
            x: Raw waveform for ``input_type="waveform"``, raw dB mel for
                ``input_type="mel"``, or already-trimmed MuQ mel features for
                ``input_type="muq_mel"``.
            attention_mask: Optional waveform-level or mel-level mask.
            input_type: ``"waveform"``, ``"mel"``, or ``"muq_mel"``. Raw mel
                input must use MuQ's mel parameters:
                ``sample_rate=24000``, ``n_fft=2048``, ``hop_length=240``,
                ``n_mels=128``, ``is_db=True``.

        Returns:
            By default, a Conformer BaseModelOutput-like object whose
            ``last_hidden_state`` is ``[batch_size, frame_count, hidden_size]``.
            If ``return_attention_mask=True``, returns
            ``(features, encoder_attention_mask)``.
        """
        hidden_states, encoder_attention_mask = self.prepare_encoder_inputs(
            x,
            attention_mask,
            input_type=input_type,
        )
        outputs = self.model.model.conformer(
            hidden_states,
            attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if return_attention_mask:
            return outputs, encoder_attention_mask
        return outputs

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pool encoder frames before task heads.

        Args:
            hidden_states: Encoder features shaped
                ``[batch_size, frame_count, hidden_size]``.
            attention_mask: Optional encoder-level mask shaped
                ``[batch_size, frame_count]``.

        Returns:
            ``[batch_size, hidden_size]`` for ``"mean"`` and ``"cls"`` pooling,
            or the unpooled ``[batch_size, frame_count, hidden_size]`` tensor
            when pooling is ``None`` or ``"none"``.
        """
        if self.pooling is None or self.pooling == "none":
            return hidden_states

        if self.pooling == "cls":
            return hidden_states[:, 0]

        if self.pooling == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)

            mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
            mask = mask.unsqueeze(-1)
            denominator = mask.sum(dim=1).clamp_min(1.0)
            return (hidden_states * mask).sum(dim=1) / denominator

        raise ValueError(f"unsupported pooling mode: {self.pooling!r}")

    @staticmethod
    def _validate_tensordict(output: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not isinstance(output, Mapping):
            raise TypeError("MuQTaskHead must return a Mapping[str, torch.Tensor]")
        if not output:
            raise ValueError("MuQTaskHead returned an empty TensorDict")
        tensor_output = {}
        for key, value in output.items():
            if not isinstance(key, str) or not key:
                raise TypeError("MuQTaskHead output keys must be non-empty strings")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"MuQTaskHead output {key!r} is not a torch.Tensor")
            tensor_output[key] = value
        return tensor_output

    def forward(
        self,
        x,
        attention_mask: torch.Tensor | None = None,
        input_type: str = "waveform",
        output_attentions: bool = False,
        output_hidden_states: bool | None = None,
        return_dict: bool = True,
        return_features: bool = False,
        **kwargs,
    ):
        """Run feature extraction, active task-head prediction, or the original MuQ path.

        Args:
            x: Raw waveform shaped ``[batch_size, timestep]`` or
                ``[batch_size, 1, timestep]`` when ``input_type="waveform"``.
                Raw dB mel shaped ``[batch_size, 128, mel_frame_count]`` or
                ``[batch_size, 1, 128, mel_frame_count]`` when
                ``input_type="mel"``.
            attention_mask: Optional waveform-level or mel-level mask.
            input_type: ``"waveform"``, ``"mel"``, or ``"muq_mel"``. For
                ``"mel"``, use ``sample_rate=24000``, ``n_fft=2048``,
                ``hop_length=240``, ``n_mels=128``, ``is_db=True``.
            return_features: In task-head mode, return
                ``(task_outputs, encoder_features)`` instead of only
                ``task_outputs``.

        Returns:
            If ``task_head`` was provided, returns the task head's
            ``dict[str, torch.Tensor]`` directly.
            If ``return_features=True``, returns
            ``(dict[str, torch.Tensor], BaseModelOutput)``.
            If no task head is provided and ``feature_only=True``, returns
            encoder features with ``last_hidden_state`` shaped
            ``[batch_size, frame_count, hidden_size]``.
            If ``feature_only=False``, delegates to the wrapped MuQ model.
        """
        if self.task_head is not None:
            features, encoder_attention_mask = self.encode(
                x,
                attention_mask=attention_mask,
                input_type=input_type,
                output_attentions=output_attentions,
                output_hidden_states=False if output_hidden_states is None else output_hidden_states,
                return_dict=return_dict,
                return_attention_mask=True,
            )
            last_hidden_state = (
                features.last_hidden_state if hasattr(features, "last_hidden_state") else features[0]
            )
            head_input = self.pool_hidden_states(last_hidden_state, encoder_attention_mask)
            with _autocast_for(head_input, self.base_dtype):
                outputs = self.task_head(head_input)
            outputs = self._validate_tensordict(outputs)
            outputs = {
                name: output.to(dtype=self.adapter_dtype) for name, output in outputs.items()
            }

            if return_features:
                return outputs, features
            return outputs

        if self.feature_only:
            return self.encode(
                x,
                attention_mask=attention_mask,
                input_type=input_type,
                output_attentions=output_attentions,
                output_hidden_states=True if output_hidden_states is None else output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        if input_type != "waveform":
            raise ValueError("non-waveform input requires feature_only=True or task_head")

        # MuQ's public forward discards its codebook logits and returns encoder
        # features. Route through the dtype-aware encoder path so FP16 base
        # weights also work for explicit non-feature-only calls.
        return self.encode(
            x,
            attention_mask=attention_mask,
            input_type=input_type,
            output_attentions=output_attentions,
            output_hidden_states=True if output_hidden_states is None else output_hidden_states,
            return_dict=return_dict,
        )
