from contextlib import nullcontext

import torch


_REDUCED_BASE_DTYPES = (torch.float16,)


def _autocast_for(x: torch.Tensor, dtype: torch.dtype):
    """Return an autocast context for reduced-precision adapter compute."""
    if dtype not in _REDUCED_BASE_DTYPES:
        return nullcontext()
    return torch.autocast(device_type=x.device.type, dtype=dtype)
