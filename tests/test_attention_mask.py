import unittest

import muq
import torch

from muqlora import MuQLoRA


MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"


def prefix_mask(lengths: tuple[int, ...], frame_count: int) -> torch.Tensor:
    values = torch.zeros(len(lengths), frame_count, dtype=torch.float32)
    for batch_index, length in enumerate(lengths):
        values[batch_index, :length] = 1
    return values


class AttentionMaskTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = muq.MuQ.from_pretrained(MODEL_ID, local_files_only=True)
        cls.model = MuQLoRA(
            base,
            r=2,
            alpha=4.0,
            target_modules=("linear_q", "linear_v"),
            num_target_layers=1,
            feature_only=True,
            drop_muq_head=True,
            base_dtype=torch.float32,
            adapter_dtype=torch.float32,
            runtime_device="cpu",
        ).eval()

    def assert_encoder_lengths(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        input_type: str,
    ) -> None:
        hidden_states, encoder_mask = self.model.prepare_encoder_inputs(
            x,
            attention_mask=attention_mask,
            input_type=input_type,
        )
        self.assertEqual(tuple(hidden_states.shape), (2, 38, 1024))
        self.assertIsNotNone(encoder_mask)
        self.assertEqual(encoder_mask.dtype, torch.bool)
        self.assertEqual(encoder_mask.sum(dim=-1).tolist(), [21, 38])

    def test_encoder_lengths_for_all_input_types(self):
        self.assert_encoder_lengths(
            torch.zeros(2, 128, 150),
            prefix_mask((83, 150), 150),
            input_type="muq_mel",
        )
        self.assert_encoder_lengths(
            torch.zeros(2, 128, 151),
            prefix_mask((84, 151), 151),
            input_type="mel",
        )
        self.assert_encoder_lengths(
            torch.zeros(2, 36_000),
            prefix_mask((19_920, 36_000), 36_000),
            input_type="waveform",
        )

    def test_padding_values_are_removed_before_frontend_convolution(self):
        attention_mask = prefix_mask((83, 150), 150)
        baseline = torch.zeros(2, 128, 150)
        changed_padding = baseline.clone()
        changed_padding[0, :, 83:] = 999

        baseline_hidden, baseline_mask = self.model.prepare_encoder_inputs(
            baseline,
            attention_mask=attention_mask,
            input_type="muq_mel",
        )
        changed_hidden, changed_mask = self.model.prepare_encoder_inputs(
            changed_padding,
            attention_mask=attention_mask,
            input_type="muq_mel",
        )

        torch.testing.assert_close(changed_hidden, baseline_hidden, rtol=0, atol=0)
        torch.testing.assert_close(changed_mask, baseline_mask, rtol=0, atol=0)

    def test_invalid_attention_masks_are_rejected(self):
        x = torch.zeros(2, 128, 150)
        valid = prefix_mask((83, 150), 150)

        with self.assertRaisesRegex(ValueError, "shape"):
            self.model.prepare_encoder_inputs(
                x,
                attention_mask=valid[:, :-1],
                input_type="muq_mel",
            )

        non_binary = valid.clone()
        non_binary[0, 0] = 0.5
        with self.assertRaisesRegex(ValueError, "only 0 and 1"):
            self.model.prepare_encoder_inputs(
                x,
                attention_mask=non_binary,
                input_type="muq_mel",
            )

        non_prefix = valid.clone()
        non_prefix[0, 1:3] = torch.tensor([0, 1])
        with self.assertRaisesRegex(ValueError, "leading prefix"):
            self.model.prepare_encoder_inputs(
                x,
                attention_mask=non_prefix,
                input_type="muq_mel",
            )

        non_finite = valid.clone()
        non_finite[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.model.prepare_encoder_inputs(
                x,
                attention_mask=non_finite,
                input_type="muq_mel",
            )


if __name__ == "__main__":
    unittest.main()
