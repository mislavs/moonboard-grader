"""
Generator for MoonBoard climbing problems using trained VAE.

This module provides functionality to generate new climbing problems
by sampling from the learned latent space of a trained Conditional VAE.
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import logging

from moonboard_core import grid_to_moves, validate_moves, decode_grade
from .vae import ConditionalVAE

logger = logging.getLogger(__name__)

# Valid parameter ranges
MIN_TEMPERATURE = 0.1
MAX_TEMPERATURE = 2.0
MIN_THRESHOLD = 0.0
MAX_THRESHOLD = 1.0


def _validate_generation_params(
    temperature: float,
    threshold: float,
    num_samples: int
) -> None:
    """
    Validate generation parameters.
    
    Args:
        temperature: Sampling temperature
        threshold: Binary conversion threshold
        num_samples: Number of samples to generate
        
    Raises:
        ValueError: If parameters are out of valid range
    """
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}, "
            f"got {temperature}"
        )
    
    if not MIN_THRESHOLD <= threshold <= MAX_THRESHOLD:
        raise ValueError(
            f"Threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}, "
            f"got {threshold}"
        )
    
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")


def _grids_from_latent(
    model: ConditionalVAE,
    z: torch.Tensor,
    grade_labels: torch.Tensor
) -> np.ndarray:
    """
    Decode latent vectors to probability grids.
    
    Args:
        model: VAE model
        z: Latent vectors (batch_size, latent_dim)
        grade_labels: Grade labels (batch_size,)
        
    Returns:
        Probability grids (batch_size, 3, 18, 11)
    """
    x_recon = model.decode(z, grade_labels)
    probs = torch.sigmoid(x_recon)
    return probs.cpu().numpy()


class ProblemGenerator:
    """
    Generator for creating new MoonBoard climbing problems.
    
    Uses a trained Conditional VAE to sample from the learned latent space
    and generate novel problems conditioned on difficulty grade.
    
    Args:
        model: Trained ConditionalVAE model
        device: Device to run generation on (cpu or cuda)
        threshold: Threshold for converting probabilities to binary holds (default: 0.5)
    """
    
    def __init__(
        self,
        model: ConditionalVAE,
        device: str = 'cpu',
        threshold: float = 0.5
    ):
        if not MIN_THRESHOLD <= threshold <= MAX_THRESHOLD:
            raise ValueError(
                f"Threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}, "
                f"got {threshold}"
            )
        
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.threshold = threshold
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cpu',
        threshold: float = 0.5
    ) -> 'ProblemGenerator':
        """
        Load generator from a saved checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
            device: Device to load model on
            threshold: Threshold for binary conversion
            
        Returns:
            ProblemGenerator instance with loaded model
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint is invalid
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False  # Allow loading model architecture
            )
            
            # Extract model config and state
            model_config = checkpoint.get('model_config', {})
            model_state = checkpoint['model_state_dict']
            
            # Infer num_grades from the actual model state if not in config
            # The grade_embedding.weight has shape (num_grades, grade_embedding_dim)
            if 'num_grades' not in model_config and 'grade_embedding.weight' in model_state:
                num_grades = model_state['grade_embedding.weight'].shape[0]
                logger.info(f"Inferred num_grades={num_grades} from checkpoint")
            else:
                num_grades = model_config.get('num_grades', 17)
            
            # Create model with same architecture
            model = ConditionalVAE(
                latent_dim=model_config.get('latent_dim', 128),
                num_grades=num_grades,
                grade_embedding_dim=model_config.get('grade_embedding_dim', 32)
            )
            
            # Load weights
            model.load_state_dict(model_state)
            
            logger.info(f"Loaded model from {checkpoint_path}")
            logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  Latent dim: {model_config.get('latent_dim', 128)}")
            
            return cls(model, device=device, threshold=threshold)
            
        except (KeyError, RuntimeError) as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        except torch.serialization.pickle.UnpicklingError as e:
            raise RuntimeError(f"Corrupted checkpoint file: {e}")
    
    def generate(
        self,
        grade_label: int,
        num_samples: int = 1,
        temperature: float = 1.0,
        validate: bool = True
    ) -> List[Dict]:
        """
        Generate new climbing problems at specified grade.
        
        Args:
            grade_label: Integer grade label (0-indexed)
            num_samples: Number of problems to generate
            temperature: Sampling temperature (higher = more random)
            validate: Whether to validate generated problems
            
        Returns:
            List of dictionaries, each containing:
            {
                'moves': List of move dictionaries,
                'grade_label': Integer grade label,
                'validation': Validation results (if validate=True)
            }
            
        Raises:
            ValueError: If parameters are invalid
        """
        # Validate parameters
        _validate_generation_params(temperature, self.threshold, num_samples)
        
        # Generate batch with same grade
        grade_labels = [grade_label] * num_samples
        return self.generate_batch(grade_labels, temperature, validate)
    
    def generate_batch(
        self,
        grade_labels: List[int],
        temperature: float = 1.0,
        validate: bool = True
    ) -> List[Dict]:
        """
        Generate multiple problems with different grades.
        
        Args:
            grade_labels: List of grade labels to generate
            temperature: Sampling temperature
            validate: Whether to validate generated problems
            
        Returns:
            List of problem dictionaries
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not grade_labels:
            return []
        
        num_samples = len(grade_labels)
        _validate_generation_params(temperature, self.threshold, num_samples)
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            z = z * temperature
            
            # Create grade labels tensor
            grades = torch.tensor(
                grade_labels,
                dtype=torch.long,
                device=self.device
            )
            
            # Decode to probability grids
            grids = _grids_from_latent(self.model, z, grades)
        
        # Convert each grid to moves
        results = []
        for i in range(num_samples):
            grid = grids[i]
            moves = grid_to_moves(grid, threshold=self.threshold)
            
            result = {
                'moves': moves,
                'grade_label': grade_labels[i]
            }
            
            if validate:
                validation = validate_moves(moves)
                result['validation'] = validation
                
                if not validation['valid']:
                    logger.warning(
                        f"Generated invalid problem at grade {grade_labels[i]}: "
                        f"{validation['errors']}"
                    )
            
            results.append(result)
        
        return results
    
    def generate_with_retry(
        self,
        grade_label: int,
        num_samples: int = 1,
        max_attempts: int = 10,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        Generate valid problems with retry logic.
        
        Keeps generating until we get the requested number of valid problems
        or reach max_attempts.
        
        Args:
            grade_label: Integer grade label
            num_samples: Number of valid problems to generate
            max_attempts: Maximum number of generation attempts
            temperature: Sampling temperature
            
        Returns:
            List of valid problem dictionaries
        """
        valid_problems = []
        attempts = 0
        
        while len(valid_problems) < num_samples and attempts < max_attempts:
            # Generate a batch
            batch_size = min(num_samples - len(valid_problems), 10)
            problems = self.generate(
                grade_label,
                num_samples=batch_size,
                temperature=temperature,
                validate=True
            )
            
            # Keep only valid problems
            for problem in problems:
                if problem['validation']['valid']:
                    valid_problems.append(problem)
                    if len(valid_problems) >= num_samples:
                        break
            
            attempts += 1
        
        if len(valid_problems) < num_samples:
            logger.warning(
                f"Only generated {len(valid_problems)}/{num_samples} valid problems "
                f"after {attempts} attempts"
            )
        
        return valid_problems
    
    def interpolate(
        self,
        problem1_grid: np.ndarray,
        problem2_grid: np.ndarray,
        grade_label: int,
        steps: int = 5
    ) -> List[Dict]:
        """
        Interpolate between two problems in latent space.
        
        Args:
            problem1_grid: Grid tensor of first problem (3, 18, 11)
            problem2_grid: Grid tensor of second problem (3, 18, 11)
            grade_label: Grade label for decoding
            steps: Number of interpolation steps
            
        Returns:
            List of interpolated problems
        """
        with torch.no_grad():
            # Encode both problems
            x1 = torch.from_numpy(problem1_grid).unsqueeze(0).to(self.device)
            x2 = torch.from_numpy(problem2_grid).unsqueeze(0).to(self.device)
            
            mu1, _ = self.model.encode(x1)
            mu2, _ = self.model.encode(x2)
            
            # Interpolate in latent space
            alphas = np.linspace(0, 1, steps)
            interpolated = []
            
            for alpha in alphas:
                z = (1 - alpha) * mu1 + alpha * mu2
                
                # Decode
                grade = torch.tensor([grade_label], dtype=torch.long, device=self.device)
                x_recon = self.model.decode(z, grade)
                probs = torch.sigmoid(x_recon)
                grid = probs[0].cpu().numpy()
                
                # Convert to moves
                moves = grid_to_moves(grid, threshold=self.threshold)
                validation = validate_moves(moves)
                
                interpolated.append({
                    'moves': moves,
                    'grade_label': grade_label,
                    'alpha': float(alpha),
                    'validation': validation
                })
        
        return interpolated


def format_problem_output(
    problem: Dict,
    include_grade: bool = False,
    grade_names: Optional[List[str]] = None
) -> Dict:
    """
    Format a generated problem for output.
    
    Args:
        problem: Problem dictionary from generator
        include_grade: Whether to include grade information
        grade_names: List of grade names for decoding labels
        
    Returns:
        Formatted problem dictionary
    """
    output = {
        'moves': problem['moves']
    }
    
    if include_grade:
        grade_label = problem['grade_label']
        if grade_names and 0 <= grade_label < len(grade_names):
            output['grade'] = grade_names[grade_label]
            output['grade_label'] = grade_label
        else:
            output['grade_label'] = grade_label
    
    # Include validation stats if available
    if 'validation' in problem:
        validation = problem['validation']
        output['stats'] = validation['stats']
        if not validation['valid']:
            output['validation_errors'] = validation['errors']
        if validation['warnings']:
            output['validation_warnings'] = validation['warnings']
    
    return output

