# MuQ-LoRA
LoRA adapter for MuQ

## Runtime precision

`MuQLoRA` keeps MuQ in BF16 and normalization modules in FP32 by default. This
is the preferred CUDA policy. For MPS inference, pass `runtime_device="mps"`
when constructing the wrapper. If the base uses BF16 or FP16, MuQLoRA emits a
warning and disables the FP32-norm policy because MPSGraph normalization
requires compatible activation and normalization-statistics dtypes.

```python
model = MuQLoRA(base_muq_model, runtime_device="mps")
model = model.to("mps")
```
