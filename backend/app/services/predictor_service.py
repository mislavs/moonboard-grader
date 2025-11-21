"""
Predictor service for managing ML model inference.

Handles model loading, prediction logic, and error handling.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from src.predictor import Predictor
from ..core.config import settings

logger = logging.getLogger(__name__)


class PredictorService:
    """
    Service class for managing the predictor model.

    Handles model lifecycle and provides prediction interface.
    """

    def __init__(
        self, model_path: Optional[Path] = None, device: Optional[str] = None
    ):
        """
        Initialize the predictor service.

        Args:
            model_path: Path to the model checkpoint.
                Defaults to settings.model_path
            device: Device for inference (cpu/cuda).
                Defaults to settings.device
        """
        self.model_path = model_path or settings.model_path
        self.device = device or settings.device
        self._predictor: Optional[Predictor] = None
        self._is_loaded = False

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
                "Please add model_for_inference.pth to the models/ directory."
            )

        logger.info(
            f"Loading model from {self.model_path} "
            f"on device {self.device}..."
        )

        try:
            self._predictor = Predictor(
                checkpoint_path=str(self.model_path),
                device=self.device
            )
            self._is_loaded = True
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._is_loaded = False
            raise

    def predict(
        self, problem: Dict[str, Any], top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Make a prediction for a climbing problem.

        Args:
            problem: Problem data with moves
            top_k: Number of top predictions to return

        Returns:
            Dictionary with prediction results

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If problem data is invalid
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            result = self._predictor.predict(
                problem=problem,
                return_top_k=top_k
            )
            return result
        except ValueError as e:
            logger.error(f"Invalid problem data: {e}")
            raise
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded and self._predictor is not None

    @property
    def predictor(self) -> Optional[Predictor]:
        """Get the underlying predictor instance."""
        return self._predictor

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
