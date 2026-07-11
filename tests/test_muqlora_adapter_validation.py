import json
import shutil
import tempfile
import unittest
from pathlib import Path

import muq
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from muqlora import MuQLoRA, MuQLoRAAdapter
from tests.test_muqlora_training import (
    LinearTensorHead,
    MODEL_ID,
    TARGET_MODULES,
)


class AlternateTensorHead(LinearTensorHead):
    pass


class MuQLoRAAdapterValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(13)

    def fresh_base(self) -> muq.MuQ:
        return muq.MuQ.from_pretrained(MODEL_ID, local_files_only=True)

    def build_model(self, target_modules=TARGET_MODULES, task_head=True) -> MuQLoRA:
        base = self.fresh_base()
        return MuQLoRA(
            base,
            task_head=(
                LinearTensorHead(base.config.encoder_dim, output_key="logits", output_dim=4)
                if task_head
                else None
            ),
            r=2,
            alpha=4.0,
            target_modules=target_modules,
            num_target_layers=1,
            base_model_name_or_path=MODEL_ID,
        )

    def save_package(self, tmpdir: str) -> Path:
        package_path = Path(tmpdir) / "adapter"
        MuQLoRAAdapter.from_model(self.build_model()).save(package_path)
        return package_path

    def copy_package_with_config_edit(self, source: Path, tmpdir: str, name: str, edit) -> Path:
        target = Path(tmpdir) / name
        shutil.copytree(source, target)
        config_path = target / "adapter_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        edit(config)
        config_path.write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    def test_constructor_and_set_adapter_reject_invalid_parameters(self):
        base = self.fresh_base()
        with self.assertRaisesRegex(TypeError, "MuQTaskHead"):
            MuQLoRA(base, task_head=nn.Linear(base.config.encoder_dim, 4))

        model = self.build_model()
        with self.assertRaisesRegex(TypeError, "MuQLoRAAdapter"):
            model.set_adapter("name")

    def test_target_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = self.save_package(tmpdir)

            mismatched_targets = self.build_model(target_modules=("linear_q",))
            adapter = MuQLoRAAdapter.load(
                package_path,
                task_head=LinearTensorHead(
                    mismatched_targets.model.config.encoder_dim,
                    output_key="logits",
                    output_dim=4,
                ),
            )
            with self.assertRaisesRegex(ValueError, "target_manifest mismatch"):
                mismatched_targets.set_adapter(adapter)

            wrapper_type_path = self.copy_package_with_config_edit(
                package_path,
                tmpdir,
                "wrapper-type",
                lambda config: config["target_manifest"][0].update(
                    {"wrapper_type": "WrongWrapper"}
                ),
            )
            wrapper_type_adapter = MuQLoRAAdapter.load(
                wrapper_type_path,
                task_head=LinearTensorHead(
                    self.build_model(task_head=False).model.config.encoder_dim,
                    output_key="logits",
                    output_dim=4,
                ),
            )
            with self.assertRaisesRegex(ValueError, "target_manifest mismatch"):
                self.build_model(task_head=False).set_adapter(wrapper_type_adapter)

            rank_path = self.copy_package_with_config_edit(
                package_path,
                tmpdir,
                "rank",
                lambda config: config["target_manifest"][0].update({"rank": 99}),
            )
            rank_adapter = MuQLoRAAdapter.load(
                rank_path,
                task_head=LinearTensorHead(
                    self.build_model(task_head=False).model.config.encoder_dim,
                    output_key="logits",
                    output_dim=4,
                ),
            )
            with self.assertRaisesRegex(ValueError, "target_manifest mismatch"):
                self.build_model(task_head=False).set_adapter(rank_adapter)

            shape_path = self.copy_package_with_config_edit(
                package_path,
                tmpdir,
                "shape",
                lambda config: config["target_manifest"][0]["tensor_shapes"].update(
                    {"lora_A.weight": [1, 1]}
                ),
            )
            shape_adapter = MuQLoRAAdapter.load(
                shape_path,
                task_head=LinearTensorHead(
                    self.build_model(task_head=False).model.config.encoder_dim,
                    output_key="logits",
                    output_dim=4,
                ),
            )
            with self.assertRaisesRegex(ValueError, "target_manifest mismatch"):
                self.build_model(task_head=False).set_adapter(shape_adapter)

    def test_tensor_key_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = self.save_package(tmpdir)

            missing_key_path = Path(tmpdir) / "missing-key"
            shutil.copytree(package_path, missing_key_path)
            missing_tensors = load_file(str(missing_key_path / "adapter_model.safetensors"))
            missing_tensors.pop(next(iter(missing_tensors)))
            save_file(missing_tensors, str(missing_key_path / "adapter_model.safetensors"))
            with self.assertRaisesRegex(ValueError, "tensor keys mismatch"):
                MuQLoRAAdapter.load(
                    missing_key_path,
                    task_head=LinearTensorHead(
                        self.build_model(task_head=False).model.config.encoder_dim,
                        output_key="logits",
                        output_dim=4,
                    ),
                )

            extra_key_path = Path(tmpdir) / "extra-key"
            shutil.copytree(package_path, extra_key_path)
            extra_tensors = load_file(str(extra_key_path / "adapter_model.safetensors"))
            extra_tensors["extra.weight"] = torch.zeros(1)
            save_file(extra_tensors, str(extra_key_path / "adapter_model.safetensors"))
            with self.assertRaisesRegex(ValueError, "tensor keys mismatch"):
                MuQLoRAAdapter.load(
                    extra_key_path,
                    task_head=LinearTensorHead(
                        self.build_model(task_head=False).model.config.encoder_dim,
                        output_key="logits",
                        output_dim=4,
                    ),
                )

    def test_legacy_registry_fields_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = self.save_package(tmpdir)
            legacy_path = self.copy_package_with_config_edit(
                package_path,
                tmpdir,
                "legacy-fields",
                lambda config: config.update(
                    {
                        "adapter_id": "old-id",
                        "adapter_name": "old-name",
                        "topology": [],
                    }
                ),
            )
            with self.assertRaisesRegex(ValueError, "unsupported fields"):
                MuQLoRAAdapter.load(
                    legacy_path,
                    task_head=LinearTensorHead(
                        self.build_model(task_head=False).model.config.encoder_dim,
                        output_key="logits",
                        output_dim=4,
                    ),
                )

    def test_task_head_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = self.save_package(tmpdir)
            hidden_size = self.build_model(task_head=False).model.config.encoder_dim

            with self.assertRaisesRegex(ValueError, "pass task_head"):
                MuQLoRAAdapter.load(package_path)
            with self.assertRaisesRegex(TypeError, "MuQTaskHead"):
                MuQLoRAAdapter.load(package_path, task_head=nn.Linear(hidden_size, 4))
            with self.assertRaisesRegex(ValueError, "task head type mismatch"):
                MuQLoRAAdapter.load(
                    package_path,
                    task_head=AlternateTensorHead(
                        hidden_size,
                        output_key="logits",
                        output_dim=4,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
