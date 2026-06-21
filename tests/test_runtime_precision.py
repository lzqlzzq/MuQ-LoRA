import unittest

import torch

from muqlora import MuQLoRA


class RuntimePrecisionPolicyTest(unittest.TestCase):
    def test_default_and_cuda_policies_preserve_fp32_norms(self):
        self.assertTrue(
            MuQLoRA.resolve_keep_norm_fp32(torch.bfloat16, True, runtime_device=None)
        )
        self.assertTrue(
            MuQLoRA.resolve_keep_norm_fp32(torch.bfloat16, True, runtime_device="cuda")
        )

    def test_mps_reduced_precision_disables_fp32_norms(self):
        with self.assertWarnsRegex(RuntimeWarning, "MPSGraph normalization"):
            self.assertFalse(
                MuQLoRA.resolve_keep_norm_fp32(
                    torch.bfloat16,
                    True,
                    runtime_device="mps",
                )
            )

    def test_mps_fp32_policy_is_unchanged(self):
        self.assertTrue(
            MuQLoRA.resolve_keep_norm_fp32(torch.float32, True, runtime_device="mps")
        )

    def test_reduced_precision_keeps_frontend_fp32_on_every_backend(self):
        self.assertEqual(
            MuQLoRA.resolve_frontend_dtype(torch.bfloat16, runtime_device="mps"),
            torch.float32,
        )
        self.assertEqual(
            MuQLoRA.resolve_frontend_dtype(torch.float16, runtime_device="mps"),
            torch.float32,
        )
        self.assertEqual(
            MuQLoRA.resolve_frontend_dtype(torch.bfloat16, runtime_device="cuda"),
            torch.float32,
        )
        self.assertEqual(
            MuQLoRA.resolve_frontend_dtype(torch.bfloat16, runtime_device=None),
            torch.float32,
        )


if __name__ == "__main__":
    unittest.main()
