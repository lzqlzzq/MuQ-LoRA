import gc
import unittest

import muq
import torch

from muqlora import MuQLoRA


MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
ACTIVATION_RTOL = 5e-2
ACTIVATION_ATOL = 5e-2
TARGET_MODULES = (
    "linear_q",
    "linear_v",
    "pointwise_conv1",
    "pointwise_conv2",
    "intermediate_dense",
    "output_dense",
)


def available_accelerator_devices() -> list[torch.device]:
    devices = []
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
    return devices


def build_neutral_wrapper(
    model: muq.MuQ,
    device: torch.device,
    base_dtype: torch.dtype,
) -> MuQLoRA:
    # LoRA B starts at zero, but fixing A too keeps the two wrapper state dicts
    # identical apart from the deliberate base-dtype conversion.
    with torch.random.fork_rng():
        torch.manual_seed(0)
        wrapper = MuQLoRA(
            model,
            r=8,
            alpha=16.0,
            target_modules=TARGET_MODULES,
            num_target_layers=12,
            feature_only=True,
            drop_muq_head=True,
            base_dtype=base_dtype,
            adapter_dtype=torch.float32,
            keep_norm_fp32=True,
            runtime_device=device,
        )
    return wrapper.to(device).eval()


def tensor_outputs(value, suffix=""):
    """Yield every tensor from a module output with a stable output suffix."""
    if isinstance(value, torch.Tensor):
        yield suffix, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from tensor_outputs(item, f"{suffix}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensor_outputs(item, f"{suffix}[{key!r}]")


def capture_leaf_activations(model: MuQLoRA):
    """Capture every tensor output from every MuQModel leaf module."""
    activations = {}
    hooks = []
    for name, module in model.model.model.named_modules():
        if name and not any(module.children()):
            def capture(_module, _inputs, output, name=name):
                for suffix, tensor in tensor_outputs(output):
                    activations.setdefault(f"{name}{suffix}", []).append(
                        tensor.detach().float().cpu()
                    )

            hooks.append(module.register_forward_hook(capture))
    return activations, hooks


def compare_activation(
    scope: str,
    name: str,
    fp32_activation: torch.Tensor,
    reduced_activation: torch.Tensor,
) -> bool:
    fp32_activation = fp32_activation.float().cpu()
    reduced_activation = reduced_activation.float().cpu()
    if fp32_activation.shape != reduced_activation.shape:
        raise AssertionError(
            f"{scope}.{name} shape mismatch: "
            f"{tuple(fp32_activation.shape)} != {tuple(reduced_activation.shape)}"
        )

    absolute_error = (fp32_activation - reduced_activation).abs()
    max_abs_error = absolute_error.max().item()
    max_rel_error = (
        absolute_error / fp32_activation.abs().clamp_min(1e-5)
    ).max().item()
    is_close = torch.allclose(
        fp32_activation,
        reduced_activation,
        rtol=ACTIVATION_RTOL,
        atol=ACTIVATION_ATOL,
    )
    print(
        f"[{scope}] layer={name} shape={tuple(fp32_activation.shape)} "
        f"allclose={is_close} max_abs={max_abs_error:.6g} "
        f"max_rel={max_rel_error:.6g}"
    )
    return is_close


class BackendActivationPrecisionTest(unittest.TestCase):
    def test_fp16_activations_match_fp32_for_every_conformer_layer(self):
        devices = available_accelerator_devices()
        if not devices:
            self.skipTest("requires an available CUDA or MPS backend")

        # A deterministic one-second A4 tone is closer to the model's audio
        # domain than a linear ramp while remaining exactly reproducible.
        sample_rate = 24_000
        frame = torch.arange(sample_rate, dtype=torch.float32) / sample_rate
        waveform = (0.5 * torch.sin(2 * torch.pi * 440 * frame)).unsqueeze(0)

        for device in devices:
            with self.subTest(device=device.type):
                # Both checkpoint loads are deterministic. Building separate
                # wrappers is required because MuQLoRA injects LoRA in-place.
                reference = build_neutral_wrapper(
                    muq.MuQ.from_pretrained(MODEL_ID), device, torch.float32
                )
                candidate = build_neutral_wrapper(
                    muq.MuQ.from_pretrained(MODEL_ID), device, torch.float16
                )

                reference_leaves, reference_hooks = capture_leaf_activations(reference)
                candidate_leaves, candidate_hooks = capture_leaf_activations(candidate)
                try:
                    with torch.no_grad():
                        fp32_output = reference(
                            waveform.to(device),
                            input_type="waveform",
                            output_hidden_states=True,
                        )
                        fp16_output = candidate(
                            waveform.to(device),
                            input_type="waveform",
                            output_hidden_states=True,
                        )
                finally:
                    for hook in reference_hooks + candidate_hooks:
                        hook.remove()

                self.assertEqual(
                    len(fp32_output.hidden_states),
                    len(fp16_output.hidden_states),
                )
                self.assertEqual(
                    reference_leaves.keys(),
                    candidate_leaves.keys(),
                )
                failures = []
                for name, fp32_activations in reference_leaves.items():
                    reduced_activations = candidate_leaves[name]
                    self.assertEqual(
                        len(fp32_activations),
                        len(reduced_activations),
                        f"leaf invocation count differs for {name}",
                    )
                    for invocation, (fp32_activation, reduced_activation) in enumerate(
                        zip(fp32_activations, reduced_activations)
                    ):
                        label = f"leaf.{name}#{invocation}"
                        if not compare_activation(
                            device.type,
                            label,
                            fp32_activation,
                            reduced_activation,
                        ):
                            failures.append(label)

                for layer_index, (fp32_activation, fp16_activation) in enumerate(
                    zip(fp32_output.hidden_states, fp16_output.hidden_states)
                ):
                    if not compare_activation(
                        device.type,
                        f"conformer.{layer_index:02d}",
                        fp32_activation,
                        fp16_activation,
                    ):
                        failures.append(f"conformer.{layer_index:02d}")

                self.assertFalse(
                    failures,
                    f"FP16 activations diverged from FP32 on {device.type} at layers {failures}; "
                    f"rtol={ACTIVATION_RTOL}, atol={ACTIVATION_ATOL}",
                )

                del fp16_output
                del fp32_output
                del candidate
                del reference
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                else:
                    torch.mps.empty_cache()


if __name__ == "__main__":
    unittest.main()
