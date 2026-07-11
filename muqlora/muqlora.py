import warnings
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

import muq
from muqlora._core.precision import (
    apply_precision_policy,
    assert_dtype_policy,
    dtype_to_name,
    install_norm_precision_hooks,
    resolve_frontend_dtype,
    resolve_keep_norm_fp32,
)
from muqlora._core.runtime import (
    MUQ_MEL_INPUT_CONFIG,
    encode as _encode,
    pool_hidden_states as _pool_hidden_states,
    prepare_encoder_inputs as _prepare_encoder_inputs,
)
from muqlora._core.targets import (
    adapter_weight_tensors,
    copy_lora_tensors,
    inject_lora_targets,
    iter_lora_modules,
    target_manifest,
    validate_adapter_manifest,
)
from muqlora.adapter import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    MuQLoRAAdapter,
    MuQLoRAConfiguration,
)
from muqlora.head import MuQTaskHead, validate_tensordict
from muqlora.module import LoRAConv1d, LoRALinear, YaRNRotaryPositionalEmbedding, _autocast_for


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
    """

    resolve_keep_norm_fp32 = staticmethod(resolve_keep_norm_fp32)
    resolve_frontend_dtype = staticmethod(resolve_frontend_dtype)

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
        self.model.requires_grad_(False)

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

        self._lora_target_names = inject_lora_targets(
            self.model,
            target_modules=self.target_modules,
            num_target_layers=num_target_layers,
            r=r,
            alpha=alpha,
            compute_dtype=base_dtype,
        )

        self._apply_precision_policy()
        self.assert_dtype_policy()
        self._install_norm_precision_hooks()
        self.train()

    def _iter_lora_modules(self):
        yield from iter_lora_modules(self.model)

    def _target_manifest(self):
        return target_manifest(self.model)

    def _adapter_configuration(self) -> MuQLoRAConfiguration:
        return MuQLoRAConfiguration(
            format_version=1,
            base_model_name_or_path=self.base_model_name_or_path,
            target_modules=list(self.target_modules),
            num_target_layers=self.num_target_layers,
            precision={
                "base_dtype": dtype_to_name(self.base_dtype),
                "adapter_dtype": dtype_to_name(self.adapter_dtype),
                "frontend_dtype": dtype_to_name(self.frontend_dtype),
                "keep_norm_fp32": self.keep_norm_fp32,
            },
            target_manifest=self._target_manifest(),
            head_config=None if self.task_head is None else self.task_head.get_config(),
        )

    def _adapter_weight_tensors(self) -> dict[str, torch.Tensor]:
        return adapter_weight_tensors(self.model, self.task_head)

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
        validate_adapter_manifest(
            adapter,
            current_manifest=self._target_manifest(),
            target_modules=self.target_modules,
            num_target_layers=self.num_target_layers,
        )

    def set_adapter(self, adapter: MuQLoRAAdapter):
        self._validate_adapter_manifest(adapter)
        copy_lora_tensors(self.model, adapter.tensors)

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

    def _apply_precision_policy(self):
        apply_precision_policy(self)

    def _install_norm_precision_hooks(self):
        install_norm_precision_hooks(self)

    def assert_dtype_policy(self):
        assert_dtype_policy(self)

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
        return _prepare_encoder_inputs(
            self,
            x,
            attention_mask=attention_mask,
            input_type=input_type,
        )

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
        return _encode(
            self,
            x,
            attention_mask=attention_mask,
            input_type=input_type,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            return_attention_mask=return_attention_mask,
        )

    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _pool_hidden_states(self, hidden_states, attention_mask=attention_mask)

    @staticmethod
    def _validate_tensordict(output):
        return validate_tensordict(output)

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
        """Run feature extraction, active task-head prediction, or the original MuQ path."""
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

        return self.encode(
            x,
            attention_mask=attention_mask,
            input_type=input_type,
            output_attentions=output_attentions,
            output_hidden_states=True if output_hidden_states is None else output_hidden_states,
            return_dict=return_dict,
        )
