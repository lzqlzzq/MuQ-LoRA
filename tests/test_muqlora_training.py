import unittest

import torch
from torch import nn

from muqlora.muqlora import LoRALinear, MuQLoRA


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        return self.linear_q(x) + self.linear_v(x)


class FakeConformerLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = FakeAttention(hidden_size)
        self.dropout = nn.Dropout(p=0.75)

    def forward(self, x):
        return x + self.dropout(self.attention(x))


class FakeConformer(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [FakeConformerLayer(hidden_size) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeMuQInner(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, out_size: int):
        super().__init__()
        self.conformer = FakeConformer(hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        return self.linear(self.conformer(x))


class FakeMuQ(nn.Module):
    def __init__(self, hidden_size: int = 8, num_layers: int = 3, out_size: int = 4):
        super().__init__()
        self.model = FakeMuQInner(hidden_size, num_layers, out_size)

    def forward(self, x):
        return self.model(x)


def train_for_steps(model: nn.Module, x: torch.Tensor, target: torch.Tensor, steps: int = 4):
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=0.05,
    )

    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


class MuQLoRATrainingTest(unittest.TestCase):
    def test_lora_steps_update_only_lora_parameters_by_default(self):
        torch.manual_seed(7)

        backbone = FakeMuQ()
        lora = MuQLoRA(
            backbone,
            r=2,
            alpha=4.0,
            target_modules=["linear_q", "linear_v"],
            num_target_layers=2,
        )

        wrapped = lora.model.model.conformer.layers[-1].attention.linear_q
        self.assertIsInstance(wrapped, LoRALinear)

        frozen_parameters_before = {
            name: parameter.detach().clone()
            for name, parameter in lora.named_parameters()
            if not parameter.requires_grad
        }
        lora_b_before = wrapped.lora_B.weight.detach().clone()

        lora.train()
        self.assertFalse(lora.model.training)
        self.assertTrue(wrapped.training)
        self.assertFalse(wrapped.module.training)
        self.assertTrue(wrapped.lora_A.training)
        self.assertTrue(wrapped.lora_B.training)
        self.assertFalse(wrapped.module.weight.requires_grad)
        self.assertFalse(lora.model.model.linear.weight.requires_grad)

        x = torch.randn(5, 8)
        target = torch.randn(5, 4)
        train_for_steps(lora, x, target)

        for name, parameter_before in frozen_parameters_before.items():
            with self.subTest(name=name):
                parameter_after = dict(lora.named_parameters())[name].detach()
                self.assertTrue(torch.equal(parameter_after, parameter_before))

        self.assertFalse(torch.equal(wrapped.lora_B.weight.detach(), lora_b_before))
        self.assertIsNone(wrapped.module.weight.grad)
        self.assertIsNone(wrapped.module.bias.grad)

    def test_muq_head_updates_only_when_explicitly_enabled(self):
        torch.manual_seed(11)

        backbone = FakeMuQ()
        original_head = backbone.model.linear.weight.detach().clone()

        lora = MuQLoRA(
            backbone,
            r=2,
            alpha=4.0,
            target_modules=["linear_q"],
            num_target_layers=1,
            train_muq_head=True,
        )

        reinitialized_head = lora.model.model.linear.weight.detach().clone()
        self.assertFalse(torch.equal(reinitialized_head, original_head))
        self.assertTrue(lora.model.model.linear.weight.requires_grad)

        x = torch.randn(5, 8)
        target = torch.randn(5, 4)
        train_for_steps(lora, x, target)

        self.assertFalse(
            torch.equal(lora.model.model.linear.weight.detach(), reinitialized_head)
        )


if __name__ == "__main__":
    unittest.main()
