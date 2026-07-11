from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def _class_path(module: nn.Module) -> str:
    return f"{module.__class__.__module__}.{module.__class__.__qualname__}"


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


def validate_tensordict(output: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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
