import unittest

import muq
import torch
from tensordict import TensorDict, TensorDictBase
from torch import nn

from muqlora import LoRAConv1d, LoRALinear, MUQ_MEL_INPUT_CONFIG, MuQLoRA, MuQTaskHead


MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
TARGET_MODULES = ("linear_q", "linear_v", "pointwise_conv1", "pointwise_conv2")


class LinearTensorHead(MuQTaskHead):
    def __init__(self, hidden_size: int, output_key: str, output_dim: int):
        super().__init__()
        self.output_key = output_key
        self.projection = nn.Linear(hidden_size, output_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "output_key": self.output_key,
                "output_dim": self.projection.out_features,
            }
        )
        return config

    def forward(self, x):
        logits = self.projection(x)
        return TensorDict({self.output_key: logits}, batch_size=logits.shape[:-1])


class DictTensorHead(MuQTaskHead):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, 4)

    def forward(self, x):
        return {"logits": self.projection(x)}


def train_task_for_steps(
    model: MuQLoRA,
    output_key: str,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int = 2,
    **forward_kwargs,
):
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.01,
    )

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(x, **forward_kwargs)[output_key]
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    return optimizer


def _gradient_spectral_norm(grad: torch.Tensor) -> float:
    grad = grad.detach().float().cpu()
    if grad.ndim == 0:
        return float(grad.abs())
    if grad.ndim == 1:
        return float(torch.linalg.vector_norm(grad))
    matrix = grad.reshape(grad.shape[0], -1)
    return float(torch.linalg.matrix_norm(matrix, ord=2))


def _print_gradient_spectral_norms(model: MuQLoRA) -> dict[str, float]:
    spectral_norms = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        value = _gradient_spectral_norm(parameter.grad)
        spectral_norms[name] = value
        print(f"grad_spectral_norm name={name} value={value:.8e}")
    return spectral_norms


def build_task_wrapper(base: muq.MuQ) -> MuQLoRA:
    return MuQLoRA(
        base,
        task_head=LinearTensorHead(
            base.config.encoder_dim,
            output_key="logits",
            output_dim=4,
        ),
        r=2,
        alpha=4.0,
        target_modules=TARGET_MODULES,
        num_target_layers=1,
        base_model_name_or_path=MODEL_ID,
    )


class MuQLoRATrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)

    def fresh_base(self) -> muq.MuQ:
        return muq.MuQ.from_pretrained(MODEL_ID, local_files_only=True)

    def test_task_head_runtime_trains_only_lora_and_head(self):
        base = self.fresh_base()
        with self.assertRaisesRegex(ValueError, "train_muq_head"):
            MuQLoRA(base, train_muq_head=True)
        with self.assertRaisesRegex(TypeError, "MuQTaskHead"):
            MuQLoRA(self.fresh_base(), task_head=nn.Linear(base.config.encoder_dim, 4))

        model = build_task_wrapper(self.fresh_base())

        lora_linear_modules = [
            module for module in model.modules() if isinstance(module, LoRALinear)
        ]
        lora_conv_modules = [
            module for module in model.modules() if isinstance(module, LoRAConv1d)
        ]
        self.assertGreaterEqual(len(lora_linear_modules), 2)
        self.assertGreaterEqual(len(lora_conv_modules), 2)
        wrapped_linear = lora_linear_modules[0]
        wrapped_conv = lora_conv_modules[0]

        self.assertIsInstance(model.model.model.linear, nn.Identity)
        model.assert_dtype_policy()

        wrapped_linear_weight_before = wrapped_linear.module.weight.detach().clone()
        wrapped_linear_bias_before = wrapped_linear.module.bias.detach().clone()
        wrapped_conv_weight_before = wrapped_conv.module.weight.detach().clone()
        linear_lora_b_before = wrapped_linear.lora_B.weight.detach().clone()
        conv_lora_b_before = wrapped_conv.lora_B.weight.detach().clone()
        head_weight_before = model.task_head.projection.weight.detach().clone()

        model.train()
        self.assertFalse(model.model.training)
        self.assertTrue(wrapped_linear.training)
        self.assertFalse(wrapped_linear.module.training)
        self.assertTrue(wrapped_linear.lora_A.training)
        self.assertTrue(wrapped_linear.lora_B.training)
        self.assertFalse(wrapped_linear.module.weight.requires_grad)
        self.assertEqual(wrapped_linear.module.weight.dtype, torch.float16)
        self.assertEqual(wrapped_linear.lora_A.weight.dtype, torch.float32)
        self.assertEqual(wrapped_linear.lora_B.weight.dtype, torch.float32)
        self.assertTrue(wrapped_conv.training)
        self.assertFalse(wrapped_conv.module.training)
        self.assertTrue(wrapped_conv.lora_A.training)
        self.assertTrue(wrapped_conv.lora_B.training)
        self.assertFalse(wrapped_conv.module.weight.requires_grad)
        self.assertEqual(wrapped_conv.module.weight.dtype, torch.float16)
        self.assertEqual(wrapped_conv.lora_A.weight.dtype, torch.float32)
        self.assertEqual(wrapped_conv.lora_B.weight.dtype, torch.float32)
        self.assertEqual(model.task_head.projection.weight.dtype, torch.float32)

        norm_tensors = []
        for module in model.model.modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)):
                norm_tensors.extend(module.parameters(recurse=False))
                norm_tensors.extend(module.buffers(recurse=False))
        self.assertTrue(norm_tensors)
        self.assertTrue(
            all(
                not tensor.is_floating_point() or tensor.dtype == torch.float32
                for tensor in norm_tensors
            )
        )

        self.assertEqual(
            MUQ_MEL_INPUT_CONFIG,
            {
                "sample_rate": 24000,
                "n_fft": 2048,
                "hop_length": 240,
                "n_mels": 128,
                "is_db": True,
            },
        )

        waveform = torch.randn(1, 24000)
        mel = model.model.model.preprocessor_melspec_2048(waveform.float())

        original_head = model.task_head
        model.task_head = DictTensorHead(model.model.config.encoder_dim).to(
            device=mel.device,
            dtype=model.adapter_dtype,
        )
        with self.assertRaisesRegex(TypeError, "TensorDict"):
            model(mel, input_type="mel")
        model.task_head = original_head

        adapter_io_dtypes = []

        def capture_adapter_io(_module, inputs, output):
            adapter_io_dtypes.append((inputs[0].dtype, output.dtype))

        linear_hook = wrapped_linear.lora_A.register_forward_hook(capture_adapter_io)
        conv_hook = wrapped_conv.lora_A.register_forward_hook(capture_adapter_io)
        try:
            waveform_output, waveform_features = model(waveform, return_features=True)
            mel_output, mel_features = model(mel, input_type="mel", return_features=True)
        finally:
            linear_hook.remove()
            conv_hook.remove()

        self.assertIsInstance(waveform_output, TensorDictBase)
        self.assertIsInstance(mel_output, TensorDictBase)
        self.assertEqual(set(waveform_output.keys()), {"logits"})
        self.assertTrue(adapter_io_dtypes)
        self.assertTrue(
            all(
                input_dtype == torch.float16 and output_dtype == torch.float16
                for input_dtype, output_dtype in adapter_io_dtypes
            )
        )
        self.assertEqual(waveform_features.last_hidden_state.dtype, torch.float16)
        self.assertEqual(mel_features.last_hidden_state.dtype, torch.float16)
        self.assertEqual(waveform_output["logits"].dtype, torch.float32)
        self.assertEqual(mel_output["logits"].dtype, torch.float32)
        torch.testing.assert_close(
            mel_features.last_hidden_state,
            waveform_features.last_hidden_state,
        )
        torch.testing.assert_close(mel_output["logits"], waveform_output["logits"])

        target = torch.randn(1, 4)
        optimizer = train_task_for_steps(model, "logits", mel, target, input_type="mel")
        grad_spectral_norms = _print_gradient_spectral_norms(model)

        self.assertTrue(grad_spectral_norms)
        self.assertTrue(any(".lora_" in name for name in grad_spectral_norms))
        self.assertTrue(any(name.startswith("task_head.") for name in grad_spectral_norms))
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(value)) and value >= 0.0
                for value in grad_spectral_norms.values()
            )
        )

        self.assertTrue(
            torch.equal(wrapped_linear.module.weight.detach(), wrapped_linear_weight_before)
        )
        self.assertTrue(
            torch.equal(wrapped_linear.module.bias.detach(), wrapped_linear_bias_before)
        )
        self.assertTrue(torch.equal(wrapped_conv.module.weight.detach(), wrapped_conv_weight_before))
        self.assertFalse(torch.equal(wrapped_linear.lora_B.weight.detach(), linear_lora_b_before))
        self.assertFalse(torch.equal(wrapped_conv.lora_B.weight.detach(), conv_lora_b_before))
        self.assertFalse(
            torch.equal(model.task_head.projection.weight.detach(), head_weight_before)
        )
        self.assertIsNone(wrapped_linear.module.weight.grad)
        self.assertIsNone(wrapped_linear.module.bias.grad)
        self.assertIsNone(wrapped_conv.module.weight.grad)
        for parameter in (
            wrapped_linear.lora_A.weight,
            wrapped_linear.lora_B.weight,
            wrapped_conv.lora_A.weight,
            wrapped_conv.lora_B.weight,
            model.task_head.projection.weight,
        ):
            self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, torch.float32)
            self.assertEqual(optimizer.state[parameter]["exp_avg_sq"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
