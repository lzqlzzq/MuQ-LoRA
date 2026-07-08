import math

import torch
from torch import nn

from muqlora.module.precision import _autocast_for


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
