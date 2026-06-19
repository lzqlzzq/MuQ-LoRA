import math
from collections.abc import Mapping, Sequence

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


class LoRALinear(nn.Module):
    def __init__(self, module: nn.Linear, r: int = 8, alpha: float = 16):
        super().__init__()

        self.module = module
        self.in_features = module.in_features
        self.out_features = module.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

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
        return self.module(x) + self.lora_B(self.lora_A(x)) * self.scaling

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

    def __init__(self, module: nn.Conv1d, r: int = 8, alpha: float = 16):
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
        return self.module(x) + self.lora_B(self.lora_A(x)) * self.scaling

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
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("r must be positive")
        if num_target_layers < 0:
            raise ValueError("num_target_layers must be non-negative")
        if train_muq_head and drop_muq_head:
            raise ValueError("train_muq_head and drop_muq_head cannot both be enabled")
        if heads is not None and not heads:
            raise ValueError("heads must contain at least one task head")
        if heads is not None and train_muq_head:
            raise ValueError("heads and train_muq_head cannot both be enabled")
        if heads is not None and feature_only is False:
            raise ValueError("heads require feature_only=True")

        self.model = model
        self.model.requires_grad_(False)  # Freeze the original model parameters before injecting LoRA.

        self.r = r
        self.alpha = alpha
        self.target_modules = tuple(target_modules or ())
        self.num_target_layers = num_target_layers
        self.train_muq_head = train_muq_head
        self.keep_base_model_eval = keep_base_model_eval
        self.heads = None if heads is None else (
            heads if isinstance(heads, nn.ModuleDict) else nn.ModuleDict(heads)
        )
        self.pooling = pooling
        self.feature_only = self.heads is not None or (
            not train_muq_head if feature_only is None else feature_only
        )
        self.drop_muq_head = self.heads is not None if drop_muq_head is None else drop_muq_head
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
                            module = LoRALinear(module, r, alpha)
                        elif isinstance(module, nn.Conv1d):
                            module = LoRAConv1d(module, r, alpha)
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

        if train_muq_head:
            with torch.no_grad():
                nn.init.trunc_normal_(self.model.model.linear.weight, std=0.02)
                nn.init.zeros_(self.model.model.linear.bias)

        self.train()
        self.model.model.linear.requires_grad_(train_muq_head)

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
        hidden_states = muq_model.conv(features["melspec_2048"])

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
            outputs = {name: head(head_input) for name, head in self.heads.items()}

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

        if output_hidden_states is not None:
            kwargs["output_hidden_states"] = output_hidden_states
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if input_type != "waveform":
            raise ValueError("non-waveform input requires feature_only=True or heads")
        return self.model(x, **kwargs)
