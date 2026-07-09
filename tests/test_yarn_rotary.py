import math
import unittest
from types import SimpleNamespace

import torch

from muqlora import YaRNRotaryPositionalEmbedding


class YaRNRotaryPositionalEmbeddingTest(unittest.TestCase):
    def build_config(self):
        return SimpleNamespace(
            hidden_size=16,
            num_attention_heads=2,
            rotary_embedding_base=10000.0,
            max_source_positions=8,
        )

    def test_factor_one_matches_standard_rope_frequencies(self):
        config = self.build_config()
        embedding = YaRNRotaryPositionalEmbedding(config, factor=1.0)

        dim = config.hidden_size // config.num_attention_heads
        expected = 1.0 / (
            config.rotary_embedding_base ** (torch.arange(0, dim, 2).float() / dim)
        )

        torch.testing.assert_close(embedding.inv_freq, expected)
        self.assertEqual(embedding.attention_factor, 1.0)

    def test_yarn_scales_attention_and_caches_by_shape_dtype_device(self):
        config = self.build_config()
        factor = 4.0
        embedding = YaRNRotaryPositionalEmbedding(config, factor=factor)
        hidden_states = torch.zeros(2, 6, config.hidden_size, dtype=torch.float32)

        positions = embedding(hidden_states)
        cached_positions = embedding(hidden_states)

        self.assertIs(positions, cached_positions)
        self.assertEqual(positions.shape, (2, 6, 1, 1, 8))
        self.assertEqual(positions.dtype, hidden_states.dtype)
        self.assertEqual(positions.device, hidden_states.device)
        self.assertAlmostEqual(
            positions[0, 0, 0, 0, 0].item(),
            0.1 * math.log(factor) + 1.0,
        )
        self.assertEqual(positions[1, 0, 0, 0, 0].item(), 0.0)

    def test_yarn_uses_fp32_math_under_outer_autocast(self):
        config = self.build_config()
        embedding = YaRNRotaryPositionalEmbedding(config, factor=4.0)
        hidden_states = torch.zeros(2, 6, config.hidden_size, dtype=torch.float16)

        with torch.autocast(device_type="cpu", dtype=torch.float16):
            positions = embedding(hidden_states)

        self.assertEqual(positions.dtype, hidden_states.dtype)
        self.assertEqual(positions.shape, (2, 6, 1, 1, 8))


if __name__ == "__main__":
    unittest.main()
