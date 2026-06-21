import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import torch
from torch import nn

import muq


MUQ_MEL_INPUT_CONFIG = {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 240,
    "n_mels": 128,
    "is_db": True,
}


_NORM_MODULE_TYPES = (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)
_AUTOCAST_DTYPES = (torch.float16, torch.bfloat16)


def _autocast_for(x: torch.Tensor, dtype: torch.dtype):
    """Return an autocast context for reduced-precision adapter compute."""
    if dtype not in _AUTOCAST_DTYPES:
        return nullcontext()
    return torch.autocast(device_type=x.device.type, dtype=dtype)


def _floating_dtype(dtype: torch.dtype) -> bool:
    return torch.empty((), dtype=dtype).is_floating_point()


class LoRALinear(nn.Module):
    def __init__(
        self,
        module: nn.Linear,
        r: int = 8,
        alpha: float = 16,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.module = module
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.compute_dtype = module.weight.dtype if compute_dtype is None else compute_dtype

        self.module.requires_grad_(False)

        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.lora_A.to(device=module.weight.device, dtype=module.weight.dtype)
        self.lora_B.to(device=module.weight.device, dtype=module.weight.dtype)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B.weight.zero_()

        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def forward(self, x):
        base_output = self.module(x)
        with _autocast_for(x, self.compute_dtype):
            lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return base_output + lora_output.to(dtype=base_output.dtype)

    def train(self, mode: bool = True):
        self.training = mode
        self.module.eval()
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        return self


class LoRAConv1d(nn.Module):
    """LoRA adapter for pointwise Conv1d modules.

    This is intended for MuQ Conformer ``pointwise_conv1`` and
    ``pointwise_conv2`` modules, where input/output tensors are shaped
    ``[batch_size, channels, frame_count]`` and the wrapped convolution has
    ``kernel_size=1``.
    """

    def __init__(
        self,
        module: nn.Conv1d,
        r: int = 8,
        alpha: float = 16,
        compute_dtype: torch.dtype | None = None,
    ):
        super().__init__()

        if module.kernel_size != (1,):
            raise ValueError("LoRAConv1d only supports pointwise Conv1d with kernel_size=1")
        if module.groups != 1:
            raise ValueError("LoRAConv1d only supports Conv1d with groups=1")

        self.module = module
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.compute_dtype = module.weight.dtype if compute_dtype is None else compute_dtype

        self.module.requires_grad_(False)

        self.lora_A = nn.Conv1d(
            self.in_channels,
            r,
            kernel_size=1,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            bias=False,
        )
        self.lora_B = nn.Conv1d(r, self.out_channels, kernel_size=1, bias=False)
        self.lora_A.to(device=module.weight.device, dtype=module.weight.dtype)
        self.lora_B.to(device=module.weight.device, dtype=module.weight.dtype)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            self.lora_B.weight.zero_()

        self.lora_A.requires_grad_(True)
        self.lora_B.requires_grad_(True)

    def forward(self, x):
        base_output = self.module(x)
        with _autocast_for(x, self.compute_dtype):
            lora_output = self.lora_B(self.lora_A(x)) * self.scaling
        return base_output + lora_output.to(dtype=base_output.dtype)

    def train(self, mode: bool = True):
        self.training = mode
        self.module.eval()
        self.lora_A.train(mode)
        self.lora_B.train(mode)
        return self


class MuQLoRA(nn.Module):
    """LoRA wrapper for MuQ's Conformer encoder.

    The default path is feature-only: it runs MuQ preprocessing, conv
    subsampling, and the Conformer encoder, while skipping MuQ's original
    codebook projection head. Passing ``heads`` turns the module into a
    multi-task model that feeds pooled encoder features to each task head.

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

    Multi-head output:
        A dict mapping task name to each head output. With ``pooling="mean"``
        or ``"cls"``, each head receives ``[batch_size, hidden_size]``.
        With ``pooling="none"``, each head receives
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
        rounding. On MPS, FP32 normalization receives an FP32 activation via
        an internal input/output bridge; CUDA uses the same public precision
        policy without this bridge.
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
        heads: Mapping[str, nn.Module] | nn.ModuleDict | None = None,
        pooling: str | None = "mean",
        drop_muq_head: bool | None = None,
        base_dtype: torch.dtype = torch.bfloat16,
        adapter_dtype: torch.dtype = torch.float32,
        keep_norm_fp32: bool = True,
        runtime_device: torch.device | str | None = None,
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
        if heads is not None and not heads:
            raise ValueError("heads must contain at least one task head")
        if heads is not None and feature_only is False:
            raise ValueError("heads require feature_only=True")
        if base_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("base_dtype must be torch.float32, torch.float16, or torch.bfloat16")
        if not _floating_dtype(adapter_dtype):
            raise ValueError("adapter_dtype must be a floating-point torch dtype")

        self.model = model
        self.model.requires_grad_(False)  # Freeze the original model parameters before injecting LoRA.

        self.r = r
        self.alpha = alpha
        self.target_modules = tuple(target_modules or ())
        self.num_target_layers = num_target_layers
        self.keep_base_model_eval = keep_base_model_eval
        self.base_dtype = base_dtype
        self.adapter_dtype = adapter_dtype
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
        self.heads = None if heads is None else (
            heads if isinstance(heads, nn.ModuleDict) else nn.ModuleDict(heads)
        )
        self.pooling = pooling
        self.feature_only = self.heads is not None or (True if feature_only is None else feature_only)
        self.drop_muq_head = self.feature_only if drop_muq_head is None else drop_muq_head
        if self.drop_muq_head and not self.feature_only:
            raise ValueError("drop_muq_head requires feature_only=True")

        if self.drop_muq_head:
            self.model.model.linear = nn.Identity()

        layer_count = num_target_layers

        # Create low-rank matrices for each linear layer in the model
        for layer in reversed(self.model.model.conformer.layers):
            if layer_count:
                for name, module in layer.named_modules():
                    if name.split(".")[-1] in self.target_modules:
                        if isinstance(module, nn.Linear):
                            module = LoRALinear(module, r, alpha, compute_dtype=base_dtype)
                        elif isinstance(module, nn.Conv1d):
                            module = LoRAConv1d(module, r, alpha, compute_dtype=base_dtype)
                        else:
                            raise TypeError(
                                f"target module {name!r} is not nn.Linear or nn.Conv1d"
                            )

                        parent = layer
                        *path, last = name.split(".")
                        for p in path:
                            parent = getattr(parent, p)
                        setattr(parent, last, module)

                layer_count -= 1

        self._apply_precision_policy()
        self.assert_dtype_policy()
        self._install_norm_precision_hooks()
        self.train()

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
        return torch.float32 if base_dtype in _AUTOCAST_DTYPES else base_dtype

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

        if self.heads is not None:
            self.heads.to(dtype=self.adapter_dtype)

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
                for parameter in module.lora_B.parameters():
                    adapter_parameter_ids.add(id(parameter))
            if self.keep_norm_fp32 and isinstance(module, _NORM_MODULE_TYPES):
                for parameter in module.parameters(recurse=False):
                    norm_parameter_ids.add(id(parameter))
                for buffer in module.buffers(recurse=False):
                    norm_buffer_ids.add(id(buffer))

        for name, parameter in self.model.named_parameters():
            if id(parameter) in adapter_parameter_ids:
                self._assert_tensor_dtype(parameter, self.adapter_dtype, f"LoRA parameter {name}")
                if not parameter.requires_grad:
                    raise AssertionError(f"LoRA parameter {name} must require gradients")
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

        if self.heads is not None:
            for name, parameter in self.heads.named_parameters():
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
        """Run feature extraction, multi-head prediction, or the original MuQ path.

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
            return_features: In multi-head mode, return
                ``(task_outputs, encoder_features)`` instead of only
                ``task_outputs``.

        Returns:
            If ``heads`` were provided, returns ``dict[str, torch.Tensor]``.
            If ``return_features=True``, returns
            ``(dict[str, torch.Tensor], BaseModelOutput)``.
            If no heads are provided and ``feature_only=True``, returns
            encoder features with ``last_hidden_state`` shaped
            ``[batch_size, frame_count, hidden_size]``.
            If ``feature_only=False``, delegates to the wrapped MuQ model.
        """
        if self.heads is not None:
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
                outputs = {name: head(head_input) for name, head in self.heads.items()}
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
            raise ValueError("non-waveform input requires feature_only=True or heads")

        # MuQ's public forward discards its codebook logits and returns encoder
        # features. Route through the dtype-aware encoder path so BF16 base
        # weights also work for explicit non-feature-only calls.
        return self.encode(
            x,
            attention_mask=attention_mask,
            input_type=input_type,
            output_attentions=output_attentions,
            output_hidden_states=True if output_hidden_states is None else output_hidden_states,
            return_dict=return_dict,
        )
