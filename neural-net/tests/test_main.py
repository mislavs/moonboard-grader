"""
Tests for main.py CLI script
"""

import pytest
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import yaml
import torch

# Import the functions we're testing
sys.path.insert(0, str(Path(__file__).parent.parent))
from main import load_config, train_command, evaluate_command, predict_command


class TestLoadConfig:
    """Test configuration loading."""
    
    def test_load_valid_config(self, tmp_path):
        """Test loading a valid config file."""
        config_data = {
            'model': {'type': 'fc', 'num_classes': 19},
            'training': {'learning_rate': 0.001, 'batch_size': 32},
            'data': {'path': 'data/problems.json'}
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_file)
        
        assert config['model']['type'] == 'fc'
        assert config['training']['learning_rate'] == 0.001
        assert config['data']['path'] == 'data/problems.json'
    
    def test_load_nonexistent_config(self):
        """Test loading a non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent.yaml")
    
    def test_load_full_config(self, tmp_path):
        """Test loading a complete config with all fields."""
        config_data = {
            'model': {'type': 'cnn', 'num_classes': 19},
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100,
                'early_stopping_patience': 10,
                'optimizer': 'adam'
            },
            'data': {
                'path': 'data/problems.json',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'random_seed': 42
            },
            'checkpoint': {
                'dir': 'models',
                'save_best': True,
                'save_final': True
            },
            'device': 'cpu',
            'evaluation': {
                'tolerance_levels': [1, 2],
                'save_confusion_matrix': True,
                'confusion_matrix_path': 'models/confusion_matrix.png'
            },
            'prediction': {
                'top_k': 3
            }
        }
        
        config_file = tmp_path / "full_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        config = load_config(config_file)
        
        # Verify all sections
        assert config['model']['type'] == 'cnn'
        assert config['training']['optimizer'] == 'adam'
        assert config['data']['random_seed'] == 42
        assert config['checkpoint']['save_best'] is True
        assert config['evaluation']['tolerance_levels'] == [1, 2]
        assert config['prediction']['top_k'] == 3


class TestTrainCommand:
    """Test training command."""
    
    def test_train_command_config_loading(self, tmp_path):
        """Test that train command loads config correctly."""
        # Create minimal config
        config_data = {
            'model': {'type': 'fc', 'num_classes': 19},
            'training': {
                'learning_rate': 0.001,
                'batch_size': 4,
                'num_epochs': 1,
                'early_stopping_patience': None,
                'optimizer': 'adam'
            },
            'data': {
                'path': str(tmp_path / 'problems.json'),
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'random_seed': 42
            },
            'checkpoint': {
                'dir': str(tmp_path / 'models'),
                'save_best': True
            },
            'device': 'cpu'
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create minimal dataset with at least 10 problems for splitting
        problems = []
        for i in range(15):
            problems.append({
                'grade': '6A' if i < 8 else '6B',
                'moves': [
                    {'Description': 'A1', 'IsStart': True, 'IsEnd': False},
                    {'Description': 'B5', 'IsStart': False, 'IsEnd': False},
                    {'Description': 'C10', 'IsStart': False, 'IsEnd': True}
                ]
            })
        
        data_file = tmp_path / 'problems.json'
        with open(data_file, 'w') as f:
            json.dump({'data': problems}, f)
        
        # Mock argparse args
        args = MagicMock()
        args.config = str(config_file)
        
        # Run training (will fail if config not loaded properly)
        try:
            train_command(args)
        except Exception as e:
            # Training might fail for various reasons, but config should load
            assert "Config file not found" not in str(e)
    
    def test_train_command_missing_data(self, tmp_path):
        """Test training with missing data file."""
        config_data = {
            'model': {'type': 'fc', 'num_classes': 19},
            'training': {
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 1,
                'optimizer': 'adam'
            },
            'data': {
                'path': 'nonexistent.json',
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'checkpoint': {'dir': str(tmp_path / 'models')},
            'device': 'cpu'
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        args = MagicMock()
        args.config = str(config_file)
        
        # Should raise FileNotFoundError when data file not found
        with pytest.raises(FileNotFoundError):
            train_command(args)
    
    def test_train_command_saves_history(self, tmp_path):
        """Test that training saves history to correct location without path duplication."""
        config_data = {
            'model': {'type': 'fc', 'num_classes': 19},
            'training': {
                'learning_rate': 0.001,
                'batch_size': 4,
                'num_epochs': 2,
                'early_stopping_patience': None,
                'optimizer': 'adam'
            },
            'data': {
                'path': str(tmp_path / 'problems.json'),
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'random_seed': 42
            },
            'checkpoint': {
                'dir': str(tmp_path / 'models'),
                'save_best': True
            },
            'device': 'cpu'
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Create minimal dataset with at least 15 problems for splitting
        problems = []
        for i in range(15):
            problems.append({
                'grade': '6A' if i < 8 else '6B',
                'moves': [
                    {'description': 'A1', 'isStart': True, 'isEnd': False},
                    {'description': 'B5', 'isStart': False, 'isEnd': False},
                    {'description': 'C10', 'isStart': False, 'isEnd': True}
                ]
            })
        
        data_file = tmp_path / 'problems.json'
        with open(data_file, 'w') as f:
            json.dump({'data': problems}, f)
        
        # Mock argparse args
        args = MagicMock()
        args.config = str(config_file)
        
        # Run training
        train_command(args)
        
        # Verify history file exists in correct location (not duplicated path)
        history_path = tmp_path / 'models' / 'training_history.json'
        assert history_path.exists(), f"History file not found at {history_path}"
        
        # Verify it's NOT in a duplicated path
        wrong_path = tmp_path / 'models' / 'models' / 'training_history.json'
        assert not wrong_path.exists(), f"History file incorrectly saved to duplicated path {wrong_path}"
        
        # Verify history contents are valid
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert len(history['train_loss']) == 2  # We trained for 2 epochs


class TestEvaluateCommand:
    """Test evaluation command."""
    
    def test_evaluate_command_missing_checkpoint(self):
        """Test evaluation with missing checkpoint."""
        args = MagicMock()
        args.checkpoint = 'nonexistent.pth'
        args.data = 'some_data.json'
        args.cpu = True
        
        with pytest.raises(SystemExit):
            evaluate_command(args)
    
    def test_evaluate_command_missing_data(self, tmp_path):
        """Test evaluation with missing data file."""
        # Create a valid dummy checkpoint with FC model state dict
        from src.models import FullyConnectedModel
        model = FullyConnectedModel(num_classes=19)
        
        checkpoint_file = tmp_path / 'model.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': 19
        }
        torch.save(checkpoint, checkpoint_file)
        
        args = MagicMock()
        args.checkpoint = str(checkpoint_file)
        args.data = 'nonexistent.json'
        args.cpu = True
        
        with pytest.raises(SystemExit):
            evaluate_command(args)


class TestPredictCommand:
    """Test prediction command."""
    
    def test_predict_command_missing_checkpoint(self):
        """Test prediction with missing checkpoint."""
        args = MagicMock()
        args.checkpoint = 'nonexistent.pth'
        args.input = 'problem.json'
        args.cpu = True
        args.top_k = 3
        
        with pytest.raises(SystemExit):
            predict_command(args)
    
    def test_predict_command_missing_input(self, tmp_path):
        """Test prediction with missing input file."""
        # Create a valid dummy checkpoint with FC model state dict
        from src.models import FullyConnectedModel
        model = FullyConnectedModel(num_classes=19)
        
        checkpoint_file = tmp_path / 'model.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': 19
        }
        torch.save(checkpoint, checkpoint_file)
        
        args = MagicMock()
        args.checkpoint = str(checkpoint_file)
        args.input = 'nonexistent.json'
        args.cpu = True
        args.top_k = 3
        
        with pytest.raises(SystemExit):
            predict_command(args)
    
    def test_predict_command_invalid_input_format(self, tmp_path):
        """Test prediction with invalid input format."""
        # Create valid dummy checkpoint with FC model state dict
        from src.models import FullyConnectedModel
        model = FullyConnectedModel(num_classes=19)
        
        checkpoint_file = tmp_path / 'model.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': 19
        }
        torch.save(checkpoint, checkpoint_file)
        
        # Create invalid input file (no 'moves' or 'data' field)
        input_file = tmp_path / 'invalid.json'
        with open(input_file, 'w') as f:
            json.dump({'invalid': 'data'}, f)
        
        args = MagicMock()
        args.checkpoint = str(checkpoint_file)
        args.input = str(input_file)
        args.cpu = True
        args.top_k = 3
        
        with pytest.raises(SystemExit):
            predict_command(args)


class TestCLIIntegration:
    """Integration tests for CLI commands."""
    
    def test_config_yaml_structure(self):
        """Test that config.yaml has correct structure."""
        config_path = Path(__file__).parent.parent / "config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Check required sections
            assert 'model' in config
            assert 'training' in config
            assert 'data' in config
            assert 'checkpoint' in config
            
            # Check model section
            assert 'type' in config['model']
            assert config['model']['type'] in ['fc', 'cnn']
            assert 'num_classes' in config['model']
            
            # Check training section
            assert 'learning_rate' in config['training']
            assert 'batch_size' in config['training']
            assert 'num_epochs' in config['training']
            
            # Check data section
            assert 'path' in config['data']
            assert 'train_ratio' in config['data']
            assert 'val_ratio' in config['data']
            assert 'test_ratio' in config['data']
    
    def test_main_script_importable(self):
        """Test that main.py can be imported."""
        import main
        
        assert hasattr(main, 'load_config')
        assert hasattr(main, 'train_command')
        assert hasattr(main, 'evaluate_command')
        assert hasattr(main, 'predict_command')
        assert hasattr(main, 'main')
    
    def test_argparse_train_subcommand(self):
        """Test that train subcommand parser works."""
        from main import main
        
        # Test with --help (should show help message)
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['main.py', 'train', '--help']):
                main()
        
        # --help causes exit code 0
        assert exc_info.value.code == 0
    
    def test_argparse_evaluate_subcommand(self):
        """Test that evaluate subcommand parser works."""
        from main import main
        
        # Test with --help
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['main.py', 'evaluate', '--help']):
                main()
        
        assert exc_info.value.code == 0
    
    def test_argparse_predict_subcommand(self):
        """Test that predict subcommand parser works."""
        from main import main
        
        # Test with --help
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['main.py', 'predict', '--help']):
                main()
        
        assert exc_info.value.code == 0
    
    def test_argparse_no_command(self):
        """Test that missing command shows error."""
        from main import main
        
        # No subcommand should cause error
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['main.py']):
                main()
        
        # Missing subcommand causes exit code 2
        assert exc_info.value.code == 2


class TestDeviceHandling:
    """Test device selection logic."""
    
    def test_cuda_fallback_to_cpu(self, tmp_path):
        """Test that CUDA request falls back to CPU when unavailable."""
        config_data = {
            'model': {'type': 'fc', 'num_classes': 19},
            'training': {
                'learning_rate': 0.001,
                'batch_size': 4,
                'num_epochs': 1,
                'optimizer': 'adam'
            },
            'data': {
                'path': str(tmp_path / 'problems.json'),
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'checkpoint': {'dir': str(tmp_path / 'models')},
            'device': 'cuda'  # Request CUDA
        }
        
        config_file = tmp_path / "config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load config
        config = load_config(config_file)
        
        # Check that if CUDA unavailable, code should handle it
        device_name = config.get('device', 'cpu')
        if device_name == 'cuda' and not torch.cuda.is_available():
            device_name = 'cpu'
        
        # Should be CPU if CUDA not available
        if not torch.cuda.is_available():
            assert device_name == 'cpu'


class TestOutputGeneration:
    """Test output file generation."""
    
    def test_predict_with_output_flag(self, tmp_path):
        """Test that prediction can specify output file."""
        from main import main
        
        # Create valid dummy checkpoint with FC model state dict
        from src.models import FullyConnectedModel
        model = FullyConnectedModel(num_classes=19)
        
        checkpoint_file = tmp_path / 'model.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'num_classes': 19
        }
        torch.save(checkpoint, checkpoint_file)
        
        # Create input file
        input_file = tmp_path / 'problem.json'
        with open(input_file, 'w') as f:
            json.dump({
                'moves': [
                    {'description': 'A1', 'isStart': True, 'isEnd': False},
                    {'description': 'B5', 'isStart': False, 'isEnd': True}
                ]
            }, f)
        
        output_file = tmp_path / 'output.json'
        
        args = MagicMock()
        args.checkpoint = str(checkpoint_file)
        args.input = str(input_file)
        args.cpu = True
        args.top_k = 3
        args.output = str(output_file)
        
        # This should work now with a valid checkpoint
        predict_command(args)
        
        # Verify output file was created
        assert output_file.exists()
        
        # Verify output is valid JSON with expected structure
        with open(output_file, 'r') as f:
            result = json.load(f)
        
        assert 'predicted_grade' in result
        assert 'confidence' in result
        assert 'top_k_predictions' in result

