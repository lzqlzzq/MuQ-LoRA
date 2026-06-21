import unittest

import muq
import torch
from torch import nn

from muqlora import LoRAConv1d, LoRALinear, MUQ_MEL_INPUT_CONFIG, MuQLoRA


MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"


def train_task_for_steps(
    model: MuQLoRA,
    task_name: str,
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
        output = model(x, **forward_kwargs)[task_name]
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()

    return optimizer


class MuQLoRAIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.base = muq.MuQ.from_pretrained(MODEL_ID)

    def test_multitask_steps_update_only_lora_and_task_head(self):
        with self.assertRaisesRegex(ValueError, "train_muq_head"):
            MuQLoRA(self.base, train_muq_head=True)

        model = MuQLoRA(
            self.base,
            heads={"genre": nn.Linear(self.base.config.encoder_dim, 4)},
            r=2,
            alpha=4.0,
            target_modules=["linear_q", "linear_v", "pointwise_conv1", "pointwise_conv2"],
            num_target_layers=1,
        )

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
        head_weight_before = model.heads["genre"].weight.detach().clone()

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
        self.assertEqual(model.heads["genre"].weight.dtype, torch.float32)

        norm_tensors = []
        for module in model.model.modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.modules.batchnorm._BatchNorm)):
                norm_tensors.extend(module.parameters(recurse=False))
                norm_tensors.extend(module.buffers(recurse=False))
        self.assertTrue(norm_tensors)
        self.assertTrue(
            all(not tensor.is_floating_point() or tensor.dtype == torch.float32 for tensor in norm_tensors)
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

        # Raw MuQ waveform input: [batch_size, timestep], 1 second at 24 kHz.
        waveform = torch.randn(1, 24000)
        # Raw MuQ mel input: [batch_size, n_mels=128, mel_frame_count].
        mel = model.model.model.preprocessor_melspec_2048(waveform.float())

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

        self.assertTrue(adapter_io_dtypes)
        self.assertTrue(
            all(input_dtype == torch.float16 and output_dtype == torch.float16
                for input_dtype, output_dtype in adapter_io_dtypes)
        )
        self.assertEqual(waveform_features.last_hidden_state.dtype, torch.float16)
        self.assertEqual(mel_features.last_hidden_state.dtype, torch.float16)
        self.assertEqual(waveform_output["genre"].dtype, torch.float32)
        self.assertEqual(mel_output["genre"].dtype, torch.float32)
        torch.testing.assert_close(
            mel_features.last_hidden_state,
            waveform_features.last_hidden_state,
        )
        torch.testing.assert_close(mel_output["genre"], waveform_output["genre"])

        target = torch.randn(1, 4)
        optimizer = train_task_for_steps(model, "genre", mel, target, input_type="mel")

        self.assertTrue(
            torch.equal(wrapped_linear.module.weight.detach(), wrapped_linear_weight_before)
        )
        self.assertTrue(
            torch.equal(wrapped_linear.module.bias.detach(), wrapped_linear_bias_before)
        )
        self.assertTrue(torch.equal(wrapped_conv.module.weight.detach(), wrapped_conv_weight_before))
        self.assertFalse(torch.equal(wrapped_linear.lora_B.weight.detach(), linear_lora_b_before))
        self.assertFalse(torch.equal(wrapped_conv.lora_B.weight.detach(), conv_lora_b_before))
        self.assertFalse(torch.equal(model.heads["genre"].weight.detach(), head_weight_before))
        self.assertIsNone(wrapped_linear.module.weight.grad)
        self.assertIsNone(wrapped_linear.module.bias.grad)
        self.assertIsNone(wrapped_conv.module.weight.grad)
        for parameter in (
            wrapped_linear.lora_A.weight,
            wrapped_linear.lora_B.weight,
            wrapped_conv.lora_A.weight,
            wrapped_conv.lora_B.weight,
            model.heads["genre"].weight,
        ):
            self.assertEqual(optimizer.state[parameter]["exp_avg"].dtype, torch.float32)
            self.assertEqual(optimizer.state[parameter]["exp_avg_sq"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
