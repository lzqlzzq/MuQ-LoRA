import unittest

import muq
import torch
from torch import nn

from muqlora import LoRALinear, MUQ_MEL_INPUT_CONFIG, MuQLoRA


MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"


def train_task_for_steps(
    model: MuQLoRA,
    task_name: str,
    x: torch.Tensor,
    target: torch.Tensor,
    steps: int = 2,
    **forward_kwargs,
):
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.05,
    )

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(x, **forward_kwargs)[task_name]
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()


class MuQLoRAIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(7)
        cls.base = muq.MuQ.from_pretrained(MODEL_ID)

    def test_multitask_steps_update_only_lora_and_task_head(self):
        model = MuQLoRA(
            self.base,
            heads={"genre": nn.Linear(self.base.config.encoder_dim, 4)},
            r=2,
            alpha=4.0,
            target_modules=["linear_q", "linear_v"],
            num_target_layers=1,
        )

        lora_modules = [module for module in model.modules() if isinstance(module, LoRALinear)]
        self.assertGreaterEqual(len(lora_modules), 2)
        wrapped = lora_modules[0]

        self.assertIsInstance(model.model.model.linear, nn.Identity)

        wrapped_weight_before = wrapped.module.weight.detach().clone()
        wrapped_bias_before = wrapped.module.bias.detach().clone()
        lora_b_before = wrapped.lora_B.weight.detach().clone()
        head_weight_before = model.heads["genre"].weight.detach().clone()

        model.train()
        self.assertFalse(model.model.training)
        self.assertTrue(wrapped.training)
        self.assertFalse(wrapped.module.training)
        self.assertTrue(wrapped.lora_A.training)
        self.assertTrue(wrapped.lora_B.training)
        self.assertFalse(wrapped.module.weight.requires_grad)

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

        waveform_output, waveform_features = model(waveform, return_features=True)
        mel_output, mel_features = model(mel, input_type="mel", return_features=True)
        torch.testing.assert_close(
            mel_features.last_hidden_state,
            waveform_features.last_hidden_state,
        )
        torch.testing.assert_close(mel_output["genre"], waveform_output["genre"])

        target = torch.randn(1, 4)
        train_task_for_steps(model, "genre", mel, target, input_type="mel")

        self.assertTrue(torch.equal(wrapped.module.weight.detach(), wrapped_weight_before))
        self.assertTrue(torch.equal(wrapped.module.bias.detach(), wrapped_bias_before))
        self.assertFalse(torch.equal(wrapped.lora_B.weight.detach(), lora_b_before))
        self.assertFalse(torch.equal(model.heads["genre"].weight.detach(), head_weight_before))
        self.assertIsNone(wrapped.module.weight.grad)
        self.assertIsNone(wrapped.module.bias.grad)


if __name__ == "__main__":
    unittest.main()
