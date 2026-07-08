# MuQ-LoRA
LoRA adapter for MuQ

## Runtime precision

`MuQLoRA` keeps MuQ in FP16 and normalization modules in FP32 by default.
BF16 is unsupported: MuQ's activation scale needs FP16's finer mantissa.
The complete convolutional frontend (Conv2d, BatchNorm, and projection) stays
FP32 on every backend, then casts its result to FP16 before the Conformer
encoder. `keep_norm_fp32=True` has the same meaning on CUDA and MPS: a reduced
precision norm input is temporarily converted to FP32 and its result is cast
back to the caller dtype.

```python
device = "mps"  # or "cuda"
model = MuQLoRA(base_muq_model).to(device)
```

## YaRN RoPE scaling

For MuQ checkpoints whose Conformer uses `position_embeddings_type="rotary"`,
`MuQLoRA` can replace the rotary positional embedding with YaRN scaling:

```python
model = MuQLoRA(
    base_muq_model,
    yarn_factor=4.0,
    yarn_original_max_position_embeddings=base_muq_model.model.conformer.config.max_source_positions,
)
```

If `yarn_original_max_position_embeddings` is omitted, MuQLoRA uses the MuQ
config's `max_position_embeddings` or `max_source_positions`. The optional
YaRN knobs `yarn_attention_factor`, `yarn_beta_fast`, `yarn_beta_slow`,
`yarn_mscale`, `yarn_mscale_all_dim`, and `yarn_truncate` mirror the standard
YaRN RoPE parameters.
