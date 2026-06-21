# MuQ-LoRA
LoRA adapter for MuQ

## Runtime precision

`MuQLoRA` keeps MuQ in BF16 and normalization modules in FP32 by default. This
is the preferred CUDA policy. Whenever the base uses BF16 or FP16, MuQLoRA
keeps the complete convolutional frontend (Conv2d, BatchNorm, and projection)
in FP32 on every backend, then casts its result to the base dtype before the
Conformer encoder. This prevents frontend BatchNorm from amplifying
reduced-precision convolution rounding. `keep_norm_fp32=True` has the same
meaning on CUDA and MPS. On MPS, MuQLoRA temporarily converts only norm inputs
to FP32 and converts their results back to the caller dtype, satisfying
MPSGraph without changing the configured policy.

```python
model = MuQLoRA(base_muq_model, runtime_device="mps")
model = model.to("mps")
```
