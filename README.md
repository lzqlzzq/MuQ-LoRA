# MuQ-LoRA

LoRA wrappers and training utilities for the MuQ Conformer encoder. The package
freezes a pretrained MuQ backbone, injects trainable LoRA adapters into selected
Conformer modules, and optionally attaches task heads on top of pooled encoder
features.

## Features

- LoRA adapters for `nn.Linear` and pointwise `nn.Conv1d` modules.
- Feature-only MuQ encoder path that skips MuQ's original codebook projection
  head.
- Optional task heads with `mean`, `cls`, or no pooling.
- Waveform, raw mel, and already-trimmed MuQ mel inputs.
- FP16 base-model execution with FP32 normalization, frontend, adapters, and
  optimizer states.
- PEFT-style adapter packages with JSON target manifests and safetensors
  weights.
- Optional YaRN scaling for rotary positional embeddings.

## Installation

```bash
pip install -e .
```

The package requires Python 3.10+ plus `torch` and `muq`.

## Quick Start

```python
import muq
import torch
from tensordict import TensorDict
from torch import nn

from muqlora import MuQLoRA, MuQTaskHead


class GenreHead(MuQTaskHead):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 4)

    def forward(self, x):
        logits = self.projection(x)
        return TensorDict({"genre_logits": logits}, batch_size=logits.shape[:-1])


base = muq.MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")

model = MuQLoRA(
    base,
    r=8,
    alpha=16.0,
    target_modules=[
        "linear_q",
        "linear_v",
        "pointwise_conv1",
        "pointwise_conv2",
    ],
    num_target_layers=2,
    task_head=GenreHead(base.config.encoder_dim),
)

waveform = torch.randn(1, 24_000)
outputs = model(waveform)
genre_logits = outputs["genre_logits"]
```

Only LoRA parameters and the optional task-head parameters require gradients.
The wrapped MuQ model stays frozen.

## Inputs and Outputs

`MuQLoRA` accepts three input modes:

- `input_type="waveform"`: raw audio shaped `[batch, timestep]` or
  `[batch, 1, timestep]`.
- `input_type="mel"`: raw dB mel features shaped `[batch, 128, frames]` or
  `[batch, 1, 128, frames]`. MuQ drops the last mel frame during preprocessing,
  so this mode applies the same `[..., :-1]` trim internally.
- `input_type="muq_mel"`: mel features already trimmed to MuQ's internal
  preprocessing output.

Raw mel tensors must use MuQ's preprocessing parameters:

```python
from muqlora import MUQ_MEL_INPUT_CONFIG

assert MUQ_MEL_INPUT_CONFIG == {
    "sample_rate": 24000,
    "n_fft": 2048,
    "hop_length": 240,
    "n_mels": 128,
    "is_db": True,
}
```

Without `task_head`, `MuQLoRA` returns Conformer encoder features. With
`task_head`, it returns that task head's TensorDict directly. Use
`return_features=True` to get both:

```python
task_outputs, features = model(waveform, return_features=True)
last_hidden_state = features.last_hidden_state
```

`task_head` must be a `MuQTaskHead`. Arbitrary `nn.Module` heads are not
supported, and task-head outputs must be `tensordict.TensorDict` instances.

## Precision Policy

By default, `MuQLoRA` uses:

- frozen MuQ weights in `base_dtype=torch.float16`;
- normalization modules in FP32 when `keep_norm_fp32=True`;
- the complete convolutional frontend in FP32;
- LoRA adapters and task heads stored in `adapter_dtype=torch.float32`;
- local autocast for adapter and task-head matmuls at the base dtype.

This policy keeps the frontend BatchNorm statistics and normalization kernels
stable while still reducing the main Conformer compute footprint.

```python
device = "cuda"  # or "mps"
model = MuQLoRA(base).to(device)
model.assert_dtype_policy()
```

BF16 is intentionally unsupported because MuQ's activation scale needs FP16's
finer mantissa.

## YaRN RoPE Scaling

For MuQ checkpoints whose Conformer uses `position_embeddings_type="rotary"`,
`MuQLoRA` can replace the rotary positional embedding with YaRN scaling:

```python
model = MuQLoRA(
    base,
    yarn_factor=4.0,
    yarn_original_max_position_embeddings=base.model.conformer.config.max_source_positions,
)
```

If `yarn_original_max_position_embeddings` is omitted, MuQLoRA uses the MuQ
config's `max_position_embeddings` or `max_source_positions`. The optional
YaRN knobs `yarn_attention_factor`, `yarn_beta_fast`, `yarn_beta_slow`,
`yarn_mscale`, `yarn_mscale_all_dim`, and `yarn_truncate` mirror the standard
YaRN RoPE parameters.

## Adapter Packages

Adapter packages are sidecar objects. `MuQLoRA` does not maintain an adapter
registry; it only carries the current adapter state.

```python
from muqlora.adapter import MuQLoRAAdapter

adapter = MuQLoRAAdapter.from_model(model)
adapter.save("genre-adapter")

loaded = MuQLoRA(
    muq.MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter"),
    target_modules=["linear_q", "linear_v", "pointwise_conv1", "pointwise_conv2"],
    num_target_layers=2,
)
loaded_adapter = MuQLoRAAdapter.load(
    "genre-adapter",
    task_head=GenreHead(loaded.model.config.encoder_dim),
)
loaded.set_adapter(loaded_adapter)
```

An adapter package contains:

- `adapter_config.json`: target manifest, tensor shapes, precision
  metadata, base model reference, and task head metadata.
- `adapter_model.safetensors`: LoRA adapter tensors and task head state only.

Loading fails if the current target module list, module types, ranks, tensor
shapes, tensor keys, or task head type do not match the saved package.

## Training Memory Estimate

For the 12-layer MuQ Conformer config (`hidden_size=1024`,
`intermediate_size=4096`, `num_attention_heads=16`) with FP16 training and
AdamW optimizer state, the parameter and optimizer constant is roughly
`4.52 GiB`.

When MuQ is running the Flash/SDPA attention path (`is_flash=true`), a practical
activation estimate is linear in the encoder sequence length:

```text
VRAM_GiB(B, S) ~= 4.52 + k * 0.5 * B * 0.001465 * S
```

Where:

- `B` is batch size.
- `S` is the Conformer encoder sequence length in frames, not raw audio
  samples. With this config's `label_rate=25`, use `S ~= 25 * audio_seconds`.
- `k` is a safety multiplier for saved autograd tensors, dropout masks,
  temporary workspaces, and allocator fragmentation. Use `k=1.3` for an
  optimistic estimate and `k=1.6~2.0` for planning.

If attention falls back to a non-Flash implementation that materializes
per-layer attention matrices, use this conservative quadratic upper bound:

```text
VRAM_GiB(B, S) ~= 4.52 + k * 0.5 * B * (0.001465 * S + 0.000001431 * S^2)
```

Passing an `attention_mask` may still allocate a `[B, 1, S, S]` mask tensor
outside the attention kernel, so very long padded batches need extra headroom
even on the Flash path.

This estimate assumes full-model FP16 + AdamW training:

```text
model weight fp16 + gradient fp16 + master weight fp32 + AdamW m/v fp32
= 16 bytes per parameter
```

For LoRA-only training, replace the constant parameter term with:

```text
frozen_base_params * 2 bytes + trainable_lora_params * 16 bytes
```

For interactive planning, open the static calculator:

```text
tools/memory_calculator.html
```

It lets you adjust batch size, sequence length, Flash vs non-Flash attention,
LoRA target modules, rank, target layers, task heads, and the safety multiplier.

## Public API

```python
from muqlora import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_WEIGHTS_NAME,
    LoRAConv1d,
    LoRALinear,
    MUQ_MEL_INPUT_CONFIG,
    MuQLoRAAdapter,
    MuQLoRAConfiguration,
    MuQLoRA,
    MuQTaskHead,
    YaRNRotaryPositionalEmbedding,
)
```

The low-level LoRA and RoPE modules live under `muqlora.module`; the top-level
imports above are kept for compatibility. Sidecar adapter APIs are also exposed
from `muqlora.adapter`, and task-head APIs are exposed from `muqlora.head`:

```python
from muqlora.adapter import MuQLoRAAdapter, MuQLoRAConfiguration
from muqlora.head import MuQTaskHead
```

## Tests

Use the `muq` conda environment when available:

```bash
/home/lzq/anaconda3/bin/conda run -n muq python -m unittest tests.test_yarn_rotary
/home/lzq/anaconda3/bin/conda run -n muq python -m unittest discover -s tests
```

`tests.test_backend_activation_precision` compares FP16 and FP32 hidden states
on available CUDA or MPS devices.
