# MuQ-LoRA
LoRA adapter for MuQ

## Runtime precision

`MuQLoRA` keeps MuQ in BF16 and normalization modules in FP32 by default. This
is the preferred CUDA policy. Whenever the base uses BF16 or FP16, MuQLoRA
keeps the complete convolutional frontend (Conv2d, BatchNorm, and projection)
in FP32 on every backend, then casts its result to the base dtype before the
Conformer encoder. This prevents frontend BatchNorm from amplifying
reduced-precision convolution rounding. For MPS inference, additionally pass
`runtime_device="mps"`; MPSGraph requires the Conformer normalization modules
to use dtypes compatible with their activations, so its FP32-norm policy is
disabled with a warning.

```python
model = MuQLoRA(base_muq_model, runtime_device="mps")
model = model.to("mps")
```
