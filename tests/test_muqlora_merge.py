import unittest

import muq
import torch
from torch import nn
from transformers import Wav2Vec2ConformerConfig

from muqlora import LoRAConv1d, LoRALinear, MuQLoRA


TARGET_MODULES = (
    "linear_q",
    "linear_v",
    "pointwise_conv1",
    "pointwise_conv2",
)
MERGE_RTOL = 5e-2
MERGE_ATOL = 5e-2
MERGE_COS_SIM_MIN = 0.999


def stringify_dict_keys(value):
    if isinstance(value, dict):
        return {
            str(key): stringify_dict_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [stringify_dict_keys(item) for item in value]
    return value


def build_random_wrapper(device: torch.device) -> MuQLoRA:
    conformer_config = Wav2Vec2ConformerConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        conv_depthwise_kernel_size=7,
        position_embeddings_type="relative",
    )
    muq_config = muq.MuQConfig(
        codebook_dim=4,
        codebook_size=16,
        conv_dim=8,
        encoder_dim=32,
        encoder_depth=2,
        mask_prob=0.0,
        stat={
            "melspec_2048_mean": 0.0,
            "melspec_2048_std": 1.0,
        },
        w2v2_config=stringify_dict_keys(conformer_config.to_dict()),
    )
    return MuQLoRA(
        muq.MuQ(muq_config),
        r=4,
        alpha=8.0,
        target_modules=TARGET_MODULES,
        num_target_layers=2,
        feature_only=True,
        drop_muq_head=True,
        base_dtype=torch.float16,
        adapter_dtype=torch.float32,
        runtime_device=device,
    ).to(device).eval()


@unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
class MuQLoRAMergeTest(unittest.TestCase):
    def test_random_muqlora_merge_precision_on_mps(self):
        torch.manual_seed(17)
        device = torch.device("mps")
        model = build_random_wrapper(device)
        snapshots = {}

        self.assertEqual(
            model._lora_target_names,
            [
                "model.conformer.layers.1.self_attn.linear_q",
                "model.conformer.layers.1.self_attn.linear_v",
                "model.conformer.layers.1.conv_module.pointwise_conv1",
                "model.conformer.layers.1.conv_module.pointwise_conv2",
                "model.conformer.layers.0.self_attn.linear_q",
                "model.conformer.layers.0.self_attn.linear_v",
                "model.conformer.layers.0.conv_module.pointwise_conv1",
                "model.conformer.layers.0.conv_module.pointwise_conv2",
            ],
        )

        for target_path, wrapper in model._iter_lora_modules():
            with torch.no_grad():
                nn.init.normal_(wrapper.lora_A.weight, mean=0.0, std=0.1)
                nn.init.normal_(wrapper.lora_B.weight, mean=0.0, std=0.1)

            base_module = wrapper.module
            if isinstance(wrapper, LoRALinear):
                lora_delta = (
                    wrapper.lora_B.weight.detach().float()
                    @ wrapper.lora_A.weight.detach().float()
                )
            else:
                lora_delta = (
                    wrapper.lora_B.weight.detach().float().flatten(1)
                    @ wrapper.lora_A.weight.detach().float().flatten(1)
                ).reshape_as(base_module.weight)

            self.assertGreater(lora_delta.abs().max().item(), 0.0)
            snapshots[target_path] = {
                "base_module": base_module,
                "weight_id": id(base_module.weight),
                "bias": (
                    None
                    if base_module.bias is None
                    else base_module.bias.detach().clone()
                ),
                "expected_weight": torch.add(
                    base_module.weight.detach().float(),
                    lora_delta,
                    alpha=wrapper.scaling,
                ).to(dtype=base_module.weight.dtype),
            }

        self.assertEqual(len(snapshots), 8)
        self.assertTrue(
            all(
                torch.count_nonzero(wrapper.lora_B.weight).item() > 0
                for _, wrapper in model._iter_lora_modules()
            )
        )

        mel = torch.randn(1, 128, 64, device=device, dtype=torch.float32)
        with torch.no_grad():
            output_before = model(
                mel,
                input_type="muq_mel",
                output_hidden_states=True,
            )
            self.assertIs(model.merge_lora(), model)
            output_after = model(
                mel,
                input_type="muq_mel",
                output_hidden_states=True,
            )

        self.assertEqual(model._lora_target_names, [])
        self.assertEqual(model._target_manifest(), [])
        self.assertFalse(
            any(
                isinstance(module, (LoRALinear, LoRAConv1d))
                for module in model.modules()
            )
        )
        self.assertFalse(
            any(
                "lora_A" in name or "lora_B" in name
                for name in model.state_dict()
            )
        )

        merged_weights = {}
        for target_path, snapshot in snapshots.items():
            merged_module = model.model.get_submodule(target_path)
            self.assertIs(merged_module, snapshot["base_module"])
            self.assertEqual(id(merged_module.weight), snapshot["weight_id"])
            self.assertEqual(merged_module.weight.device.type, "mps")
            self.assertEqual(merged_module.weight.dtype, torch.float16)
            self.assertFalse(merged_module.weight.requires_grad)
            torch.testing.assert_close(
                merged_module.weight,
                snapshot["expected_weight"],
                rtol=0,
                atol=0,
            )
            if snapshot["bias"] is not None:
                torch.testing.assert_close(
                    merged_module.bias,
                    snapshot["bias"],
                    rtol=0,
                    atol=0,
                )
            merged_weights[target_path] = merged_module.weight.detach().clone()

        before = torch.cat(
            [hidden_state.float().flatten() for hidden_state in output_before.hidden_states]
        )
        after = torch.cat(
            [hidden_state.float().flatten() for hidden_state in output_after.hidden_states]
        )
        absolute_error = (before - after).abs()
        max_abs_error = absolute_error.max().item()
        mean_abs_error = absolute_error.mean().item()
        max_rel_error = (
            absolute_error / before.abs().clamp_min(1e-5)
        ).max().item()
        relative_l2_error = (
            torch.linalg.vector_norm(before - after)
            / torch.linalg.vector_norm(before).clamp_min(1e-12)
        ).item()
        cosine_similarity = torch.nn.functional.cosine_similarity(
            before,
            after,
            dim=0,
            eps=1e-12,
        ).item()
        print(
            "merge_precision "
            f"device=mps max_abs={max_abs_error:.8e} "
            f"mean_abs={mean_abs_error:.8e} "
            f"max_rel={max_rel_error:.8e} "
            f"rel_l2={relative_l2_error:.8e} "
            f"cos_sim={cosine_similarity:.10f}"
        )

        torch.testing.assert_close(
            after,
            before,
            rtol=MERGE_RTOL,
            atol=MERGE_ATOL,
        )
        self.assertGreaterEqual(cosine_similarity, MERGE_COS_SIM_MIN)
        model.assert_dtype_policy()

        self.assertIs(model.merge_lora(), model)
        for target_path, expected_weight in merged_weights.items():
            torch.testing.assert_close(
                model.model.get_submodule(target_path).weight,
                expected_weight,
                rtol=0,
                atol=0,
            )


if __name__ == "__main__":
    unittest.main()
