"""Tests for generator module."""

import pytest
import torch
import numpy as np
from pathlib import Path
import shutil
from uuid import uuid4

from src.vae import ConditionalVAE
from src.generator import ProblemGenerator, format_problem_output
from src.label_space import EvaluationLabelContext

TMP_ROOT = Path(".tmp_pytest_sandbox")
TMP_ROOT.mkdir(exist_ok=True)


@pytest.fixture
def checkpoint_tmp_dir():
    tmp_dir = TMP_ROOT / f"tmp_{uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestProblemGenerator:
    """Test ProblemGenerator class."""
    
    @pytest.fixture
    def model(self):
        """Create a simple VAE model for testing."""
        return ConditionalVAE(
            latent_dim=32,
            num_grades=5,
            grade_embedding_dim=16
        )
    
    @pytest.fixture
    def generator(self, model):
        """Create a generator with test model."""
        return ProblemGenerator(model, device='cpu', threshold=0.5)
    
    def test_initialization(self, model):
        """Test generator initialization."""
        generator = ProblemGenerator(model, device='cpu')
        
        assert generator.model is not None
        assert generator.device == 'cpu'
        assert generator.threshold == 0.5
        assert generator.model.training is False  # Should be in eval mode
    
    def test_generate_single_problem(self, generator):
        """Test generating a single problem."""
        problems = generator.generate(
            grade_label=2,
            num_samples=1,
            validate=True
        )
        
        assert len(problems) == 1
        problem = problems[0]
        
        # Check structure
        assert 'moves' in problem
        assert 'grade_label' in problem
        assert 'validation' in problem
        assert problem['grade_label'] == 2
        
        # Check moves structure
        moves = problem['moves']
        assert isinstance(moves, list)
        if len(moves) > 0:
            move = moves[0]
            assert 'description' in move
            assert 'isStart' in move
            assert 'isEnd' in move
    
    def test_generate_multiple_problems(self, generator):
        """Test generating multiple problems."""
        problems = generator.generate(
            grade_label=1,
            num_samples=5,
            validate=True
        )
        
        assert len(problems) == 5
        
        # All should have same grade
        for problem in problems:
            assert problem['grade_label'] == 1
    
    def test_generate_batch_different_grades(self, generator):
        """Test generating problems with different grades."""
        grade_labels = [0, 1, 2, 3, 4]
        
        problems = generator.generate_batch(
            grade_labels=grade_labels,
            validate=True
        )
        
        assert len(problems) == 5
        
        # Check that grades match
        for i, problem in enumerate(problems):
            assert problem['grade_label'] == grade_labels[i]
    
    def test_temperature_parameter(self, generator):
        """Test that temperature parameter works."""
        # This just tests that it runs without error
        # Actual effect is hard to test deterministically
        problems_low = generator.generate(
            grade_label=1,
            num_samples=2,
            temperature=0.5
        )
        
        problems_high = generator.generate(
            grade_label=1,
            num_samples=2,
            temperature=1.5
        )
        
        assert len(problems_low) == 2
        assert len(problems_high) == 2
    
    def test_threshold_parameter(self):
        """Test that threshold affects hold detection."""
        model = ConditionalVAE(latent_dim=32, num_grades=5, grade_embedding_dim=16)
        
        # Generator with low threshold (more holds)
        gen_low = ProblemGenerator(model, device='cpu', threshold=0.3)
        
        # Generator with high threshold (fewer holds)
        gen_high = ProblemGenerator(model, device='cpu', threshold=0.7)
        
        problems_low = gen_low.generate(grade_label=1, num_samples=1)
        problems_high = gen_high.generate(grade_label=1, num_samples=1)
        
        # Can't guarantee which has more, but they should both run
        assert len(problems_low) == 1
        assert len(problems_high) == 1
    
    def test_generate_without_validation(self, generator):
        """Test generating without validation."""
        problems = generator.generate(
            grade_label=1,
            num_samples=2,
            validate=False
        )
        
        assert len(problems) == 2
        
        # Should not have validation key
        for problem in problems:
            assert 'validation' not in problem
    
    def test_generate_with_retry(self, generator):
        """Test generate_with_retry method."""
        # This might not produce valid problems with random model
        # but should at least attempt generation
        problems = generator.generate_with_retry(
            grade_label=1,
            num_samples=2,
            max_attempts=3,
            temperature=1.0
        )
        
        # Should return a list (possibly empty if no valid problems)
        assert isinstance(problems, list)
        assert len(problems) <= 2
    
    def test_interpolate(self, generator):
        """Test interpolation between two problems."""
        # Create two simple problem grids
        grid1 = np.zeros((3, 18, 11), dtype=np.float32)
        grid1[0, 0, 0] = 1.0  # A1 start
        grid1[2, 17, 10] = 1.0  # K18 end
        
        grid2 = np.zeros((3, 18, 11), dtype=np.float32)
        grid2[0, 0, 5] = 1.0  # F1 start
        grid2[2, 17, 5] = 1.0  # F18 end
        
        interpolated = generator.interpolate(
            problem1_grid=grid1,
            problem2_grid=grid2,
            grade_label=1,
            steps=3
        )
        
        assert len(interpolated) == 3
        
        # Check structure
        for i, problem in enumerate(interpolated):
            assert 'moves' in problem
            assert 'grade_label' in problem
            assert 'alpha' in problem
            assert 'validation' in problem
            assert 0.0 <= problem['alpha'] <= 1.0
    
    def test_from_checkpoint(self, model, checkpoint_tmp_dir):
        """Test loading generator from checkpoint."""
        checkpoint_path = checkpoint_tmp_dir / "test_checkpoint.pth"

        # Save a checkpoint
        checkpoint = {
            'epoch': 10,
            'model_state_dict': model.state_dict(),
            'model_config': {
                'latent_dim': 32,
                'num_grades': 5,
                'grade_embedding_dim': 16,
                'dropout_rate': 0.25,
            }
        }

        torch.save(checkpoint, checkpoint_path)

        # Load generator from checkpoint
        generator = ProblemGenerator.from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device='cpu'
        )

        assert generator is not None
        assert generator.model.latent_dim == 32
        assert generator.model.dropout_rate == pytest.approx(0.25)
        assert generator.label_context.label_space_mode == "global_legacy"

    def test_from_checkpoint_missing_dropout_rate_defaults_to_zero(
        self, model, checkpoint_tmp_dir
    ):
        """Legacy checkpoints without model_config.dropout_rate should still load."""
        checkpoint_path = checkpoint_tmp_dir / "legacy_no_dropout_checkpoint.pth"
        checkpoint = {
            "epoch": 10,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "latent_dim": 32,
                "num_grades": 5,
                "grade_embedding_dim": 16,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        generator = ProblemGenerator.from_checkpoint(
            checkpoint_path=str(checkpoint_path),
            device="cpu",
        )

        assert generator.model.dropout_rate == pytest.approx(0.0)

    def test_from_checkpoint_legacy_encoder_shape_mismatch_has_clear_error(
        self, checkpoint_tmp_dir
    ):
        """Legacy unconditioned-encoder checkpoints should fail with guidance."""
        model = ConditionalVAE(latent_dim=32, num_grades=5, grade_embedding_dim=16)
        checkpoint_path = checkpoint_tmp_dir / "legacy_encoder_checkpoint.pth"

        legacy_state = model.state_dict()
        legacy_in_features = model.encoder_output_size
        legacy_state["fc_mu.weight"] = (
            legacy_state["fc_mu.weight"][:, :legacy_in_features].clone()
        )
        legacy_state["fc_logvar.weight"] = (
            legacy_state["fc_logvar.weight"][:, :legacy_in_features].clone()
        )

        checkpoint = {
            "epoch": 1,
            "model_state_dict": legacy_state,
            "model_config": {
                "latent_dim": 32,
                "num_grades": 5,
                "grade_embedding_dim": 16,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(
            RuntimeError, match="legacy CVAE architecture without encoder grade conditioning"
        ):
            ProblemGenerator.from_checkpoint(str(checkpoint_path), device="cpu")

    def test_from_checkpoint_legacy_decoder_shape_mismatch_has_clear_error(
        self, checkpoint_tmp_dir
    ):
        """Legacy interpolating-decoder checkpoints should fail with guidance."""
        model = ConditionalVAE(latent_dim=32, num_grades=5, grade_embedding_dim=16)
        checkpoint_path = checkpoint_tmp_dir / "legacy_decoder_checkpoint.pth"

        legacy_state = model.state_dict()
        legacy_state["decoder.0.weight"] = (
            legacy_state["decoder.0.weight"][:, :, :3, :3].clone()
        )
        legacy_state["output_adjust.weight"] = torch.randn(3, 3, 1, 1)
        legacy_state["output_adjust.bias"] = torch.randn(3)

        checkpoint = {
            "epoch": 1,
            "model_state_dict": legacy_state,
            "model_config": {
                "latent_dim": 32,
                "num_grades": 5,
                "grade_embedding_dim": 16,
            },
        }
        torch.save(checkpoint, checkpoint_path)

        with pytest.raises(
            RuntimeError, match="legacy CVAE decoder architecture"
        ):
            ProblemGenerator.from_checkpoint(str(checkpoint_path), device="cpu")

    def test_from_checkpoint_explicit_remapped_metadata(self, checkpoint_tmp_dir):
        """New checkpoints should preserve explicit remapped label-space metadata."""
        model = ConditionalVAE(latent_dim=32, num_grades=1, grade_embedding_dim=16)
        checkpoint_path = checkpoint_tmp_dir / "test_remapped_checkpoint.pth"
        checkpoint = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "latent_dim": 32,
                "num_grades": 1,
                "grade_embedding_dim": 16,
            },
            "label_space_mode": "remapped",
            "grade_offset": 2,
            "min_grade_index": 2,
            "max_grade_index": 2,
        }
        torch.save(checkpoint, checkpoint_path)

        generator = ProblemGenerator.from_checkpoint(str(checkpoint_path), device="cpu")
        assert generator.label_context.label_space_mode == "remapped"
        assert generator.label_context.global_to_model_label(2) == 0
        assert generator.label_context.model_to_global_label(0) == 2

    def test_from_checkpoint_infers_remapped_legacy_metadata(self, checkpoint_tmp_dir):
        """Legacy compact checkpoints should be inferred as remapped."""
        model = ConditionalVAE(latent_dim=32, num_grades=3, grade_embedding_dim=16)
        checkpoint_path = checkpoint_tmp_dir / "test_legacy_remapped_checkpoint.pth"
        checkpoint = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "latent_dim": 32,
                "num_grades": 3,
                "grade_embedding_dim": 16,
            },
            "grade_offset": 2,
            "min_grade_index": 2,
            "max_grade_index": 4,
        }
        torch.save(checkpoint, checkpoint_path)

        generator = ProblemGenerator.from_checkpoint(str(checkpoint_path), device="cpu")
        assert generator.label_context.label_space_mode == "remapped"
        assert generator.label_context.global_to_model_label(2) == 0
        assert generator.label_context.global_to_model_label(4) == 2

    def test_from_checkpoint_infers_global_legacy_metadata(self, checkpoint_tmp_dir):
        """Legacy non-compact checkpoints should stay global_legacy."""
        model = ConditionalVAE(latent_dim=32, num_grades=5, grade_embedding_dim=16)
        checkpoint_path = checkpoint_tmp_dir / "test_legacy_global_checkpoint.pth"
        checkpoint = {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "model_config": {
                "latent_dim": 32,
                "num_grades": 5,
                "grade_embedding_dim": 16,
            },
            "grade_offset": 2,
            "min_grade_index": 2,
            "max_grade_index": 4,
        }
        torch.save(checkpoint, checkpoint_path)

        generator = ProblemGenerator.from_checkpoint(str(checkpoint_path), device="cpu")
        assert generator.label_context.label_space_mode == "global_legacy"
        assert generator.label_context.global_to_model_label(2) == 2
    
    def test_from_checkpoint_missing_file(self):
        """Test that missing checkpoint raises error."""
        with pytest.raises(FileNotFoundError):
            ProblemGenerator.from_checkpoint(
                checkpoint_path="nonexistent.pth",
                device='cpu'
            )


class TestFormatProblemOutput:
    """Test format_problem_output function."""
    
    def test_basic_formatting(self):
        """Test basic problem formatting."""
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'K18', 'isStart': False, 'isEnd': True}
            ],
            'grade_label': 5
        }
        
        output = format_problem_output(problem, include_grade=False)
        
        assert 'moves' in output
        assert len(output['moves']) == 2
        assert 'grade' not in output  # Not included
    
    def test_include_grade(self):
        """Test formatting with grade included."""
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False}
            ],
            'grade_label': 2
        }
        
        grade_names = ['5+', '6A', '6A+', '6B', '6B+']
        
        output = format_problem_output(
            problem,
            include_grade=True,
            grade_names=grade_names
        )
        
        assert 'grade' in output
        assert output['grade'] == '6A+'
        assert output['grade_label'] == 2
    
    def test_include_validation_errors(self):
        """Test that validation errors are included."""
        problem = {
            'moves': [],
            'grade_label': 1,
            'validation': {
                'valid': False,
                'errors': ['No holds', 'No start hold', 'No end hold'],
                'warnings': [],
                'stats': {
                    'total_holds': 0,
                    'start_holds': 0,
                    'middle_holds': 0,
                    'end_holds': 0
                }
            }
        }
        
        output = format_problem_output(problem)
        
        assert 'validation_errors' in output
        assert len(output['validation_errors']) == 3
    
    def test_include_validation_warnings(self):
        """Test that validation warnings are included."""
        problem = {
            'moves': [
                {'description': 'A1', 'isStart': True, 'isEnd': False},
                {'description': 'K18', 'isStart': False, 'isEnd': True}
            ],
            'grade_label': 1,
            'validation': {
                'valid': True,
                'errors': [],
                'warnings': ['Problem has very few holds (2)'],
                'stats': {
                    'total_holds': 2,
                    'start_holds': 1,
                    'middle_holds': 0,
                    'end_holds': 1
                }
            }
        }
        
        output = format_problem_output(problem)
        
        assert 'validation_warnings' in output
        assert len(output['validation_warnings']) == 1

    def test_include_grade_with_label_context_maps_to_global(self):
        """When label_context is provided, include-grade should emit global label/name."""
        problem = {
            "moves": [{"description": "A1", "isStart": True, "isEnd": False}],
            "grade_label": 0,
        }
        context = EvaluationLabelContext(
            label_space_mode="remapped",
            grade_offset=2,
            min_grade_index=2,
            max_grade_index=2,
            num_model_grades=1,
        )

        output = format_problem_output(problem, include_grade=True, label_context=context)
        assert output["grade"] == "6A+"
        assert output["grade_label"] == 2
        assert output["model_grade_label"] == 0

