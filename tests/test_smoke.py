"""Source and CLI checks that need no GPU dependencies or model downloads."""

import ast
import contextlib
from dataclasses import asdict
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from plw.cli import MODELS, build_parser, preflight, resolve_config, stage2_eval_argv
from plw.utils.paths import adapter_output_identity


class SourceSmokeTests(unittest.TestCase):
    def test_source_syntax(self):
        for path in (ROOT / "plw").rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                compile(path.read_text(), str(path), "exec")

    def test_local_imports_are_present(self):
        for path in (ROOT / "plw").rglob("*.py"):
            relative = path.relative_to(ROOT).with_suffix("")
            package = ".".join(relative.parts[:-1])
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if node.level:
                    module = importlib.util.resolve_name(
                        "." * node.level + module, package
                    )
                if not module.startswith("plw"):
                    continue
                target = ROOT.joinpath(*module.split("."))
                with self.subTest(path=path, module=module):
                    self.assertTrue(
                        target.with_suffix(".py").is_file() or target.is_dir()
                    )

    def test_all_main_configs_resolve(self):
        self.assertEqual(MODELS, ("bagel", "omnigen2"))
        for stage in ("stage1", "stage2"):
            for model in MODELS:
                config = resolve_config(stage, model)
                self.assertEqual(config.model_type, model)
                self.assertTrue(asdict(config))

    def test_single_trigger_presets(self):
        bagel = resolve_config("stage2", "bagel")
        omni = resolve_config("stage2", "omnigen2")
        self.assertEqual((bagel.max_train_steps, omni.max_train_steps), (1750, 2000))
        self.assertEqual(
            (bagel.contrastive_weight, omni.contrastive_weight), (5.0, 1.0)
        )
        self.assertFalse(bagel.early_stopping)
        self.assertTrue(omni.early_stopping)
        for cfg in (bagel, omni):
            self.assertEqual(cfg.image_size, 512)
            self.assertEqual(cfg.delta_limit_weight, 0.5)
            self.assertFalse(cfg.preservation_timestep_weighting)
            self.assertEqual(cfg.trigger_set_fraction, 0.5)
            self.assertEqual(cfg.trigger, "watercolor")
            self.assertEqual(cfg.trigger_set, "")
            self.assertFalse(cfg.image_cache_read_only)
            self.assertTrue(cfg.align_image_cache_with_combined_prompts)

    def test_overrides(self):
        cfg = resolve_config(
            "stage2", "bagel", overrides=["learning_rate=0.0002", "early_stopping=true"]
        )
        self.assertEqual(cfg.learning_rate, 0.0002)
        self.assertTrue(cfg.early_stopping)
        with self.assertRaisesRegex(ValueError, "Unknown configuration"):
            resolve_config("stage2", "bagel", overrides=["misspelled_option=1"])
        with self.assertRaisesRegex(ValueError, "512px"):
            resolve_config("stage2", "bagel", overrides=["image_size=1024"])
        with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps=1"):
            resolve_config(
                "stage2", "bagel", overrides=["gradient_accumulation_steps=2"]
            )
        cfg = resolve_config(
            "stage2", "bagel", overrides=["align_image_cache_with_combined_prompts=false"]
        )
        self.assertFalse(cfg.align_image_cache_with_combined_prompts)

    def test_stage1_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "example.png"
            image.touch()  # Preflight checks paths, not pixel content.
            cfg = resolve_config(
                "stage1", "bagel", overrides=[f"train_path={tmp}", f"val_path={tmp}"]
            )
            preflight("stage1", cfg)

    def test_eval_arguments(self):
        args = build_parser().parse_args(
            [
                "stage2",
                "eval",
                "--model",
                "bagel",
                "--run-dir",
                "runs/example",
                "--checkpoint",
                "1750",
            ]
        )
        argv = stage2_eval_argv(args)
        self.assertIn("msgs=2_insertion=0_None", argv)
        self.assertIn("1750", argv)
        args.trigger_override = "alternative wording"
        self.assertIn("--trigger-override", stage2_eval_argv(args))

    def test_absolute_checkpoint_never_redirects_output(self):
        self.assertEqual(adapter_output_identity("/read/only/model"), Path("model"))
        with self.assertRaises(ValueError):
            adapter_output_identity("/read/only/model", "/wrong/output")

    def test_cli_dry_runs_without_torch(self):
        for model in MODELS:
            for stage in ("stage1", "stage2"):
                proc = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "plw",
                        stage,
                        "train",
                        "--model",
                        model,
                        "--dry-run",
                    ],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                self.assertEqual(json.loads(proc.stdout)["model_type"], model)


if __name__ == "__main__":
    unittest.main()
