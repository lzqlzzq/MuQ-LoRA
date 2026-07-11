import tempfile
import unittest
from pathlib import Path

import muq
import torch
from safetensors.torch import load_file

from muqlora import MuQLoRA, MuQLoRAAdapter
from tests.test_muqlora_training import (
    LinearTensorHead,
    MODEL_ID,
    TARGET_MODULES,
    train_task_for_steps,
)


class MuQLoRAAdapterAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(11)

    def fresh_base(self) -> muq.MuQ:
        return muq.MuQ.from_pretrained(MODEL_ID, local_files_only=True)

    def build_model(self) -> MuQLoRA:
        base = self.fresh_base()
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

    def test_adapter_sidecar_save_load_and_set_round_trip(self):
        model = self.build_model()
        waveform = torch.randn(1, 24000)
        mel = model.model.model.preprocessor_melspec_2048(waveform.float())
        target = torch.randn(1, 4)
        train_task_for_steps(model, "logits", mel, target, input_type="mel")

        model.eval()
        with torch.no_grad():
            expected_output = model(mel, input_type="mel")

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = Path(tmpdir) / "adapter"
            adapter = MuQLoRAAdapter.from_model(model)
            adapter.save(package_path)

            tensor_keys = set(load_file(str(package_path / "adapter_model.safetensors")).keys())
            self.assertTrue(tensor_keys)
            self.assertTrue(all(key.startswith(("lora.", "head.")) for key in tensor_keys))
            self.assertFalse(any(".module." in key for key in tensor_keys))

            loaded = MuQLoRA(
                self.fresh_base(),
                r=2,
                alpha=4.0,
                target_modules=TARGET_MODULES,
                num_target_layers=1,
                base_model_name_or_path=MODEL_ID,
            )
            loaded_adapter = MuQLoRAAdapter.load(
                package_path,
                task_head=LinearTensorHead(
                    loaded.model.config.encoder_dim,
                    output_key="logits",
                    output_dim=4,
                ),
            )
            loaded.set_adapter(loaded_adapter)

            loaded.eval()
            with torch.no_grad():
                actual_output = loaded(mel, input_type="mel")
            torch.testing.assert_close(actual_output["logits"], expected_output["logits"])

            exported_again = loaded.export_adapter()
            self.assertEqual(
                exported_again.configuration.to_dict(),
                loaded_adapter.configuration.to_dict(),
            )


if __name__ == "__main__":
    unittest.main()
