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


def capture_frontend_activations(model: MuQLoRA):
    """Capture the Conv2dSubsampling leaf activations for one forward pass."""
    activations = {}
    hooks = []
    for name, module in model.model.model.conv.named_modules():
        if name and not any(module.children()):
            def capture(_module, _inputs, output, name=name):
                if isinstance(output, torch.Tensor):
                    activations[name] = output.detach().float().cpu()

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
    def test_bf16_activations_match_fp32_for_every_conformer_layer(self):
        devices = available_accelerator_devices()
        if not devices:
            self.skipTest("requires an available CUDA or MPS backend")

        # A deterministic one-second waveform keeps eval-mode activations stable.
        waveform = torch.linspace(-0.5, 0.5, 24_000).unsqueeze(0)

        for device in devices:
            with self.subTest(device=device.type):
                # Both checkpoint loads are deterministic. Building separate
                # wrappers is required because MuQLoRA injects LoRA in-place.
                reference = build_neutral_wrapper(
                    muq.MuQ.from_pretrained(MODEL_ID), device, torch.float32
                )
                candidate = build_neutral_wrapper(
                    muq.MuQ.from_pretrained(MODEL_ID), device, torch.bfloat16
                )

                reference_frontend, reference_hooks = capture_frontend_activations(reference)
                candidate_frontend, candidate_hooks = capture_frontend_activations(candidate)
                try:
                    with torch.no_grad():
                        fp32_output = reference(
                            waveform.to(device),
                            input_type="waveform",
                            output_hidden_states=True,
                        )
                        bf16_output = candidate(
                            waveform.to(device),
                            input_type="waveform",
                            output_hidden_states=True,
                        )
                finally:
                    for hook in reference_hooks + candidate_hooks:
                        hook.remove()

                self.assertEqual(
                    len(fp32_output.hidden_states),
                    len(bf16_output.hidden_states),
                )
                self.assertEqual(
                    reference_frontend.keys(),
                    candidate_frontend.keys(),
                )
                failures = []
                for name, fp32_activation in reference_frontend.items():
                    if not compare_activation(
                        device.type,
                        f"frontend.{name}",
                        fp32_activation,
                        candidate_frontend[name],
                    ):
                        failures.append(f"frontend.{name}")

                for layer_index, (fp32_activation, bf16_activation) in enumerate(
                    zip(fp32_output.hidden_states, bf16_output.hidden_states)
                ):
                    if not compare_activation(
                        device.type,
                        f"conformer.{layer_index:02d}",
                        fp32_activation,
                        bf16_activation,
                    ):
                        failures.append(f"conformer.{layer_index:02d}")

                self.assertFalse(
                    failures,
                    f"BF16 activations diverged from FP32 on {device.type} at layers {failures}; "
                    f"rtol={ACTIVATION_RTOL}, atol={ACTIVATION_ATOL}",
                )

                del bf16_output
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
