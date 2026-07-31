"""CLI boundary tests for the frozen classifier experiment workflow."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.commands import setup_parsers
from src.cli.evaluate import evaluate_command
from src.cli.evaluate_diagnostic import evaluate_diagnostic_command
from src.cli.predict import predict_command
from src.cli.train import deprecated_train_command, train_command
from src.cli.utils import load_config, setup_device


class TestLoadConfig:
    def test_load_valid_config(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "model": {"type": "fc", "num_classes": 19},
                    "training": {"learning_rate": 0.001},
                    "data": {"path": "problems.json"},
                }
            ),
            encoding="utf-8",
        )
        config = load_config(path)
        assert config["model"]["type"] == "fc"
        assert config["training"]["learning_rate"] == 0.001

    def test_missing_config(self):
        with pytest.raises(FileNotFoundError):
            load_config("missing.yaml")


class TestCommandRegistry:
    def _parser(self):
        parser = argparse.ArgumentParser()
        setup_parsers(parser)
        return parser

    def test_official_workflow_commands_are_registered(self):
        parser = self._parser()
        commands = [
            ["create-manifest", "--output", "manifest.json"],
            ["cross-validate", "--manifest", "manifest.json"],
            ["refit", "--manifest", "manifest.json", "--cv-report", "report.json"],
            ["evaluate", "--checkpoint", "model.pth", "--manifest", "manifest.json"],
            ["evaluate-diagnostic", "--checkpoint", "model.pth", "--data", "external.json"],
            ["predict", "--checkpoint", "model.pth", "--input", "problem.json"],
        ]
        for arguments in commands:
            assert callable(parser.parse_args(arguments).func)

    def test_train_is_a_hard_deprecation(self):
        args = self._parser().parse_args(["train", "--config", "config.yaml"])
        assert args.func is deprecated_train_command
        assert train_command is deprecated_train_command
        with pytest.raises(RuntimeError, match="evaluated the test set on every run"):
            args.func(args)

    def test_official_evaluate_does_not_accept_data(self):
        parser = self._parser()
        with pytest.raises(SystemExit):
            parser.parse_args(
                ["evaluate", "--checkpoint", "model.pth", "--data", "problems.json"]
            )


class TestEvaluationBoundaries:
    def test_locked_evaluate_rejects_legacy_checkpoint(self, tmp_path, monkeypatch):
        from src.cli import evaluate as evaluate_module
        from src.models import FullyConnectedModel

        monkeypatch.setattr(
            evaluate_module,
            "require_clean_revision",
            lambda: {"commit": "abc", "dirty": False},
        )
        checkpoint = tmp_path / "legacy.pth"
        torch.save({"model_state_dict": FullyConnectedModel(19).state_dict()}, checkpoint)
        args = MagicMock(
            checkpoint=str(checkpoint),
            manifest=str(tmp_path / "manifest.json"),
            cpu=True,
            output=None,
        )
        with pytest.raises(ValueError, match="schema v2"):
            evaluate_command(args)

    def test_diagnostic_report_is_non_comparable(self, tmp_path):
        from src.models import FullyConnectedModel

        checkpoint = tmp_path / "legacy.pth"
        torch.save({"model_state_dict": FullyConnectedModel(19).state_dict()}, checkpoint)
        data = tmp_path / "problems.json"
        data.write_text(
            json.dumps(
                {
                    "data": [
                        {
                            "grade": "6A",
                            "moves": [
                                {"description": "A1", "isStart": True, "isEnd": False},
                                {"description": "C10", "isStart": False, "isEnd": True},
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output = tmp_path / "diagnostic.json"
        evaluate_diagnostic_command(
            MagicMock(checkpoint=str(checkpoint), data=str(data), cpu=True, output=str(output))
        )
        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["evaluation_kind"] == "diagnostic"
        assert report["comparable"] is False


class TestPredictCommand:
    def test_prediction_output_remains_backward_compatible(self, tmp_path):
        from src.models import FullyConnectedModel

        checkpoint = tmp_path / "model.pth"
        torch.save({"model_state_dict": FullyConnectedModel(19).state_dict()}, checkpoint)
        problem = tmp_path / "problem.json"
        problem.write_text(
            json.dumps(
                {
                    "moves": [
                        {"description": "A1", "isStart": True, "isEnd": False},
                        {"description": "B5", "isStart": False, "isEnd": True},
                    ]
                }
            ),
            encoding="utf-8",
        )
        output = tmp_path / "prediction.json"
        predict_command(
            MagicMock(
                checkpoint=str(checkpoint),
                input=str(problem),
                cpu=True,
                top_k=3,
                output=str(output),
            )
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        assert "predicted_grade" in result
        assert "confidence" in result
        assert "top_k_predictions" in result


class TestConfigAndMain:
    def test_main_sets_cublas_determinism_before_cli_imports(self):
        source = (Path(__file__).parent.parent / "main.py").read_text(encoding="utf-8")
        setting = source.index('os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")')
        command_import = source.index("from src.cli.commands import setup_parsers")
        assert setting < command_import

    def test_config_uses_official_split_and_scheduler_schema(self):
        path = Path(__file__).parent.parent / "config.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert config["experiment"]["cv_folds"] == 5
        assert config["experiment"]["cv_seeds"] == [42, 43, 44]
        assert config["experiment"]["outer_test_ratio"] == 0.15
        assert config["training"]["deterministic"] is True
        assert config["training"]["scheduler"]["type"] == "cosine"
        assert "train_ratio" not in config["data"]

    def test_main_requires_a_command(self):
        from main import main

        with pytest.raises(SystemExit) as error:
            with patch("sys.argv", ["main.py"]):
                main()
        assert error.value.code == 2

    def test_main_dispatches_registered_function(self, monkeypatch):
        from main import main

        called = []
        monkeypatch.setattr(
            "sys.argv",
            ["main.py", "train"],
        )
        with pytest.raises(SystemExit) as error:
            main()
        assert error.value.code == 1

    def test_device_cpu(self):
        device, name = setup_device("cpu")
        assert str(device) == "cpu"
        assert name == "cpu"


class TestCLIASCIISafety:
    def test_cli_print_statements_are_ascii_safe(self):
        cli_dir = Path(__file__).parent.parent / "src" / "cli"
        files = list(cli_dir.glob("*.py")) + [Path(__file__).parent.parent / "main.py"]
        for path in files:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "print(" in stripped or "print_section_header(" in stripped or "print_completion_message(" in stripped:
                    assert all(ord(character) < 128 for character in line), (
                        f"{path.name}:{line_number} contains non-ASCII output text"
                    )
