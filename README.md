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
