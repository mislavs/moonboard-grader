"""
Generator service for managing VAE-based problem generation.

Handles model loading, problem generation logic, and error handling.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import sys

from ..core.config import settings

logger = logging.getLogger(__name__)


def _ensure_generator_in_path():
    """Ensure generator module is in sys.path."""
    generator_path = Path(__file__).parent.parent.parent.parent / "generator"
    if generator_path.exists() and str(generator_path) not in sys.path:
        sys.path.insert(0, str(generator_path))


def _get_generator_classes():
    """Import and return generator classes (lazy import)."""
    _ensure_generator_in_path()
    from src.generator import ProblemGenerator, format_problem_output
    return ProblemGenerator, format_problem_output


def _get_grade_functions():
    """Import and return grade encoding functions (lazy import)."""
    from moonboard_core import encode_grade, decode_grade, remap_label
    return encode_grade, decode_grade, remap_label


class GeneratorService:
    """
    Service class for managing the generator model.
    
    Handles model lifecycle and provides problem generation interface.
    """
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the generator service.
        
        Args:
            model_path: Path to the model checkpoint. Defaults to models/generator_model.pth
            device: Device for inference (cpu/cuda). Defaults to settings.device
        """
        self.model_path = model_path or Path("models/generator_model.pth")
        self.device = device or settings.device
        self._generator: Optional[Any] = None  # ProblemGenerator instance
        self._is_loaded = False
        self._min_grade_index: int = 0  # For filtered models, the offset for label remapping
        
    def load_model(self) -> None:
        """
        Load the model from disk.
        
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please add generator_model.pth to the models/ directory."
            )
        
        logger.info(f"Loading generator model from {self.model_path} on device {self.device}...")
        
        try:
            import torch
            
            # Load checkpoint to extract data config
            checkpoint = torch.load(
                str(self.model_path),
                map_location=self.device,
                weights_only=False
            )
            
            # Extract min_grade_index for label remapping (for filtered models)
            # Check both 'data_config' and 'config.data' for compatibility
            if 'data_config' in checkpoint:
                data_config = checkpoint['data_config']
            elif 'config' in checkpoint and 'data' in checkpoint['config']:
                data_config = checkpoint['config']['data']
            else:
                data_config = {}
            
            # Get min_grade_index from config, or infer from model architecture
            if 'min_grade_index' in data_config:
                self._min_grade_index = data_config['min_grade_index']
            else:
                # Infer num_grades from model state dict (grade_embedding.weight shape)
                model_state = checkpoint.get('model_state_dict', {})
                if 'grade_embedding.weight' in model_state:
                    num_grades = model_state['grade_embedding.weight'].shape[0]
                    logger.info(f"Inferred num_grades={num_grades} from model state")
                else:
                    # Fallback to model_config
                    model_config = checkpoint.get('model_config', {})
                    num_grades = model_config.get('num_grades', 17)
                
                if num_grades == 1:
                    # Single grade model - likely 6A+ only (index 2)
                    self._min_grade_index = 2
                    logger.info("Detected single-grade model, assuming 6A+ only (min_grade_index=2)")
                else:
                    self._min_grade_index = 0
            
            logger.info(f"Model min_grade_index={self._min_grade_index}")
            
            ProblemGenerator, _ = _get_generator_classes()
            self._generator = ProblemGenerator.from_checkpoint(
                checkpoint_path=str(self.model_path),
                device=self.device
            )
            self._is_loaded = True
            logger.info("Generator model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            self._is_loaded = False
            raise
    
    def generate_problem(
        self,
        grade: str,
        temperature: float = 1.0,
        max_attempts: int = 10
    ) -> Dict[str, Any]:
        """
        Generate a valid climbing problem at the specified grade.
        
        Args:
            grade: Font grade string (e.g., "6A+", "7B")
            temperature: Sampling temperature (higher = more random)
            max_attempts: Maximum number of attempts to generate a valid problem
            
        Returns:
            Dictionary with generated problem data including moves and metadata
            
        Raises:
            RuntimeError: If model is not loaded
            ValueError: If grade is invalid or no valid problem generated
        """
        if not self.is_loaded:
            raise RuntimeError("Generator model not loaded. Call load_model() first.")
        
        try:
            # Import grade encoding functions
            encode_grade, decode_grade, remap_label = _get_grade_functions()
            _, format_problem_output = _get_generator_classes()
            
            # Encode grade to full label index
            full_grade_label = encode_grade(grade)
            
            # Remap to filtered model index if needed
            model_grade_label = remap_label(full_grade_label, self._min_grade_index)
            
            logger.info(
                f"Generating problem at grade {grade} "
                f"(full label {full_grade_label}, model label {model_grade_label})"
            )
            
            # Generate with retry to ensure valid problem
            problems = self._generator.generate_with_retry(
                grade_label=model_grade_label,
                num_samples=1,
                max_attempts=max_attempts,
                temperature=temperature
            )
            
            if not problems:
                raise ValueError(
                    f"Failed to generate valid problem after {max_attempts} attempts"
                )
            
            # Format the output
            problem = problems[0]
            formatted = format_problem_output(problem, include_grade=True)
            
            # Add the grade string
            formatted['grade'] = grade
            
            logger.info(
                f"Successfully generated problem with {len(formatted['moves'])} moves "
                f"at grade {grade}"
            )
            
            return formatted
            
        except ValueError as e:
            logger.error(f"Invalid grade or generation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Problem generation failed: {e}")
            raise
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded and self._generator is not None
    
    @property
    def generator(self) -> Optional[Any]:
        """Get the underlying generator instance (ProblemGenerator)."""
        return self._generator
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_path": str(self.model_path),
            "device": self.device,
            "model_exists": self.model_path.exists(),
            "is_loaded": self.is_loaded
        }

