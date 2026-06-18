import math
from collections.abc import Sequence

import torch
from torch import nn

import muq


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
    def __init__(
        self,
        model: muq.MuQ,
        r: int = 8,
        alpha: float = 16.0,
        target_modules: Sequence[str] | None = None,
        num_target_layers: int = 2,
        train_muq_head: bool = False,
        keep_base_model_eval: bool = True,
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("r must be positive")
        if num_target_layers < 0:
            raise ValueError("num_target_layers must be non-negative")

        self.model = model
        self.model.requires_grad_(False)  # Freeze the original model parameters before injecting LoRA.

        self.r = r
        self.alpha = alpha
        self.target_modules = tuple(target_modules or ())
        self.num_target_layers = num_target_layers
        self.train_muq_head = train_muq_head
        self.keep_base_model_eval = keep_base_model_eval

        layer_count = num_target_layers

        # Create low-rank matrices for each linear layer in the model
        for layer in reversed(self.model.model.conformer.layers):
            if layer_count:
                for name, module in layer.named_modules():
                    if name.split(".")[-1] in self.target_modules:
                        if not isinstance(module, nn.Linear):
                            raise TypeError(f"target module {name!r} is not nn.Linear")

                        module = LoRALinear(module, r, alpha)

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
                if isinstance(module, LoRALinear):
                    module.train(mode)
        return self

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
