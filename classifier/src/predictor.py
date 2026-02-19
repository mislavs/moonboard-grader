"""
Inference Interface Module

Provides easy-to-use prediction interface for loading trained models
and making predictions on new Moonboard problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Optional
from pathlib import Path
import numpy as np

from moonboard_core.grade_encoder import decode_grade, get_num_grades
from moonboard_core.grid_builder import create_grid_tensor


class Predictor:
    """
    Predictor class for making grade predictions on Moonboard problems.

    Loads a trained model from a checkpoint and provides methods for
    single and batch predictions.

    Attributes:
        model: Loaded PyTorch model
        device: Device to run inference on (cpu or cuda)
        checkpoint_path: Path to the loaded checkpoint file
    """

    def __init__(self, checkpoint_path: Union[str, Path], device: str = "cpu"):
        """
        Initialize the Predictor by loading a trained model.

        Args:
            checkpoint_path: Path to the model checkpoint file (.pth)
            device: Device to use for inference ('cpu' or 'cuda')

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If device is invalid or checkpoint is malformed
            RuntimeError: If model loading fails

        Examples:
            >>> predictor = Predictor('models/best_model.pth')
            >>> predictor = Predictor('models/best_model.pth', device='cuda')
        """
        # Validate device
        if device not in ["cpu", "cuda"]:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")

        # Check CUDA availability
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available")

        self.device = device
        self.checkpoint_path = Path(checkpoint_path)

        # Check if checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load the checkpoint
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Validate checkpoint structure
        if "model_state_dict" not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'model_state_dict'")

        # Load filtering metadata (if present)
        self.grade_offset = checkpoint.get("grade_offset", 0)
        self.min_grade_index = checkpoint.get("min_grade_index", 0)
        self.max_grade_index = checkpoint.get("max_grade_index", 18)

        # Infer model architecture from state dict
        state_dict = checkpoint["model_state_dict"]
        self.model = self._infer_model_architecture(state_dict)

        # Load model weights
        try:
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        # Move model to device and set to eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

    def _infer_model_architecture(self, state_dict: Dict) -> nn.Module:
        """
        Infer model architecture from state dictionary.

        Args:
            state_dict: Model state dictionary

        Returns:
            Instantiated model with correct architecture

        Raises:
            ValueError: If architecture cannot be inferred
        """
        # Import model classes dynamically to avoid circular dependencies
        from .models import FullyConnectedModel, ConvolutionalModel, create_model
        from .advanced_models import ResidualCNN, DeepResidualCNN

        # Determine number of classes from final layer
        num_classes = None

        # Try common final layer names
        for key in ["fc3.weight", "network.7.weight", "fc.weight", "classifier.weight"]:
            if key in state_dict:
                num_classes = state_dict[key].shape[0]
                break

        if num_classes is None:
            num_classes = get_num_grades()

        # Infer architecture based on state dict keys
        state_keys = set(state_dict.keys())

        # Check for residual blocks (ResidualCNN or DeepResidualCNN)
        if any("res" in key for key in state_keys):
            # Check if it's deep version (has res3)
            if any("res3" in key for key in state_keys):
                return DeepResidualCNN(num_classes=num_classes)
            else:
                return ResidualCNN(num_classes=num_classes)

        # Check for attention mechanisms (also ResidualCNN)
        if any("att" in key for key in state_keys):
            return ResidualCNN(num_classes=num_classes)

        # Check for sequential network (FullyConnectedModel)
        if any("network." in key for key in state_keys):
            return FullyConnectedModel(num_classes=num_classes)

        # Check for convolutional layers (ConvolutionalModel)
        if "conv1.weight" in state_keys and "fc3.weight" in state_keys:
            return ConvolutionalModel(num_classes=num_classes)

        # Default fallback - try to create a basic model
        raise ValueError(
            "Cannot infer model architecture from state dict. "
            "Unknown model type. Please ensure the checkpoint is from a supported model."
        )

    def predict(self, problem: Dict, return_top_k: int = 3) -> Dict:
        """
        Make prediction on a single Moonboard problem.

        Args:
            problem: Problem dictionary with 'moves' key
            return_top_k: Number of top predictions to return (default: 3)

        Returns:
            Dictionary containing:
                - 'predicted_grade': Most likely grade (string)
                - 'predicted_label': Most likely grade label (int)
                - 'confidence': Probability of predicted grade (0-1)
                - 'all_probabilities': Dict mapping grades to probabilities
                - 'top_k_predictions': List of (grade, probability) tuples

        Raises:
            ValueError: If problem is invalid or missing required fields

        Examples:
            >>> problem = {'moves': [{'description': 'A1', 'isStart': True, ...}]}
            >>> result = predictor.predict(problem)
            >>> print(f"Predicted grade: {result['predicted_grade']}")
            >>> print(f"Confidence: {result['confidence']:.2%}")
        """
        # Validate input
        if not isinstance(problem, dict):
            raise ValueError("Problem must be a dictionary")
        if "moves" not in problem:
            raise ValueError("Problem must contain 'moves' key")

        # Convert problem to tensor
        try:
            tensor = create_grid_tensor(problem["moves"])
        except Exception as e:
            raise ValueError(f"Failed to process problem: {e}")

        # Add batch dimension and convert to torch tensor
        tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)

        # Make prediction
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)

        # Get predicted class and confidence
        confidence, predicted_label = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_label = predicted_label.item()

        # Convert to numpy for easier manipulation
        probs_array = probabilities.cpu().numpy()[0]

        # Unmap labels if using filtered model
        from moonboard_core.grade_encoder import unmap_label, get_filtered_grade_names

        if self.grade_offset > 0:
            # Model outputs 0-10 (for 11 classes), need to map back to original indices
            unmapped_label = unmap_label(predicted_label, self.grade_offset)
            grade_names = get_filtered_grade_names(
                self.min_grade_index, self.max_grade_index
            )
        else:
            unmapped_label = predicted_label
            grade_names = None  # Will use get_all_grades() in decode_grade

        # Get all probabilities as dict (using filtered or all grades)
        all_probs = {}
        for model_label, prob in enumerate(probs_array):
            if self.grade_offset > 0:
                original_label = unmap_label(model_label, self.grade_offset)
                grade = decode_grade(original_label)
            else:
                grade = decode_grade(model_label)
            all_probs[grade] = float(prob)

        # Get top-k predictions
        top_k_indices = np.argsort(probs_array)[-return_top_k:][::-1]
        top_k_predictions = []
        for idx in top_k_indices:
            if self.grade_offset > 0:
                original_label = unmap_label(int(idx), self.grade_offset)
                grade = decode_grade(original_label)
            else:
                grade = decode_grade(int(idx))
            top_k_predictions.append((grade, float(probs_array[idx])))

        return {
            "predicted_grade": decode_grade(unmapped_label),
            "predicted_label": unmapped_label,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "top_k_predictions": top_k_predictions,
        }

    def predict_with_attention(self, problem: Dict, return_top_k: int = 3) -> Dict:
        """
        Make prediction and return attention map for visualization.

        Extracts block3 spatial attention weights, which highlight
        the regions the model considers most important for grading.

        Args:
            problem: Problem dictionary with 'moves' key
            return_top_k: Number of top predictions to return (default: 3)

        Returns:
            Dictionary containing all fields from predict() plus:
                - 'attention_map': 2D list [18][11] of attention weights (0-1)
                  representing importance of each grid position

        Raises:
            ValueError: If problem is invalid or missing required fields

        Examples:
            >>> result = predictor.predict_with_attention(problem)
            >>> attention = result['attention_map']  # 18x11 grid
            >>> print(f"Top-left attention: {attention[0][0]:.3f}")
        """
        # Validate input
        if not isinstance(problem, dict):
            raise ValueError("Problem must be a dictionary")
        if "moves" not in problem:
            raise ValueError("Problem must contain 'moves' key")

        # Check if model supports attention extraction by checking for required attributes
        # This avoids isinstance issues with different import paths
        has_attention = (
            hasattr(self.model, "use_attention")
            and self.model.use_attention
            and hasattr(self.model, "att3_spatial")
        )

        if not has_attention:
            # Fall back to regular prediction with empty attention
            result = self.predict(problem, return_top_k)
            result["attention_map"] = None
            return result

        # Convert problem to tensor
        try:
            tensor = create_grid_tensor(problem["moves"])
        except Exception as e:
            raise ValueError(f"Failed to process problem: {e}")

        # Add batch dimension and convert to torch tensor
        tensor = torch.FloatTensor(tensor).unsqueeze(0).to(self.device)

        # Forward pass with attention extraction
        with torch.no_grad():
            x = tensor

            # Block 1: (batch, 3, 18, 11) -> (batch, 64, 9, 5)
            x = F.relu(self.model.bn1(self.model.conv1(x)))
            x = self.model.res1(x)
            x = self.model.att1_channel(x)
            x = self.model.att1_spatial(x)
            x = self.model.pool1(x)

            # Block 2: (batch, 64, 9, 5) -> (batch, 128, 4, 2)
            x = F.relu(self.model.bn2(self.model.conv2(x)))
            x = self.model.res2(x)
            x = self.model.att2_channel(x)
            x = self.model.att2_spatial(x)
            x = self.model.pool2(x)

            # Block 3: (batch, 128, 4, 2) -> (batch, 256, 4, 2)
            x = F.relu(self.model.bn3(self.model.conv3(x)))
            x = self.model.att3_channel(x)

            # Extract block3 spatial attention weights BEFORE applying
            att3_weights = self.model.att3_spatial.attention(x)  # (1, 1, 4, 2)
            x = x * att3_weights  # Apply attention

            # Upsample attention to original grid size (18x11)
            att_upsampled = F.interpolate(
                att3_weights, size=(18, 11), mode="bilinear", align_corners=False
            )
            attention_map = att_upsampled.squeeze().cpu().numpy().tolist()

            # Continue forward pass to get prediction
            x = self.model.pool_global(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.model.fc1(x))
            x = self.model.dropout1(x)
            x = F.relu(self.model.fc2(x))
            x = self.model.dropout2(x)
            logits = self.model.fc3(x)
            probabilities = F.softmax(logits, dim=1)

        # Get predicted class and confidence
        confidence, predicted_label = torch.max(probabilities, dim=1)
        confidence = confidence.item()
        predicted_label = predicted_label.item()

        # Convert to numpy for easier manipulation
        probs_array = probabilities.cpu().numpy()[0]

        # Unmap labels if using filtered model
        from moonboard_core.grade_encoder import unmap_label

        if self.grade_offset > 0:
            unmapped_label = unmap_label(predicted_label, self.grade_offset)
        else:
            unmapped_label = predicted_label

        # Get all probabilities as dict
        all_probs = {}
        for model_label, prob in enumerate(probs_array):
            if self.grade_offset > 0:
                original_label = unmap_label(model_label, self.grade_offset)
                grade = decode_grade(original_label)
            else:
                grade = decode_grade(model_label)
            all_probs[grade] = float(prob)

        # Get top-k predictions
        top_k_indices = np.argsort(probs_array)[-return_top_k:][::-1]
        top_k_predictions = []
        for idx in top_k_indices:
            if self.grade_offset > 0:
                original_label = unmap_label(int(idx), self.grade_offset)
                grade = decode_grade(original_label)
            else:
                grade = decode_grade(int(idx))
            top_k_predictions.append((grade, float(probs_array[idx])))

        return {
            "predicted_grade": decode_grade(unmapped_label),
            "predicted_label": unmapped_label,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "top_k_predictions": top_k_predictions,
            "attention_map": attention_map,
        }

    def predict_batch(self, problems: List[Dict], return_top_k: int = 3) -> List[Dict]:
        """
        Make predictions on a batch of Moonboard problems.

        Args:
            problems: List of problem dictionaries, each with 'moves' key
            return_top_k: Number of top predictions to return per problem

        Returns:
            List of prediction dictionaries (one per problem)

        Raises:
            ValueError: If problems list is empty or invalid

        Examples:
            >>> problems = [
            ...     {'moves': [...]},
            ...     {'moves': [...]}
            ... ]
            >>> results = predictor.predict_batch(problems)
            >>> for i, result in enumerate(results):
            ...     print(f"Problem {i}: {result['predicted_grade']}")
        """
        if not isinstance(problems, list):
            raise ValueError("Problems must be a list")
        if len(problems) == 0:
            raise ValueError("Problems list cannot be empty")

        # Process each problem individually, capturing per-item errors
        # instead of aborting the entire batch on first failure.
        results = []
        for i, problem in enumerate(problems):
            try:
                result = self.predict(problem, return_top_k=return_top_k)
                results.append(result)
            except Exception as e:
                results.append({
                    "error": str(e),
                    "index": i,
                })

        return results

    def predict_from_tensor(
        self, tensor: Union[np.ndarray, torch.Tensor], return_top_k: int = 3
    ) -> Dict:
        """
        Make prediction from a pre-processed tensor.

        Useful when you already have the grid tensor representation.

        Args:
            tensor: Grid tensor of shape (3, 18, 11) or (batch, 3, 18, 11)
            return_top_k: Number of top predictions to return

        Returns:
            Prediction dictionary (or list if batch input)

        Raises:
            ValueError: If tensor shape is invalid

        Examples:
            >>> tensor = create_grid_tensor(moves)
            >>> result = predictor.predict_from_tensor(tensor)
        """
        # Convert to torch tensor if needed
        if isinstance(tensor, np.ndarray):
            tensor = torch.FloatTensor(tensor)

        # Add batch dimension if single sample
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        # Validate shape
        if tensor.ndim != 4 or tensor.shape[1:] != (3, 18, 11):
            raise ValueError(
                f"Invalid tensor shape: {tensor.shape}. "
                f"Expected (batch, 3, 18, 11) or (3, 18, 11)"
            )

        # Move to device
        tensor = tensor.to(self.device)

        # Make predictions
        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = F.softmax(logits, dim=1)

        # Process each sample in batch
        results = []
        probs_array = probabilities.cpu().numpy()

        # Import for unmapping
        from moonboard_core.grade_encoder import unmap_label, get_filtered_grade_names

        for sample_probs in probs_array:
            # Get predicted class and confidence
            predicted_label = int(np.argmax(sample_probs))
            confidence = float(sample_probs[predicted_label])

            if self.grade_offset > 0:
                # Model outputs 0-10 (for 11 classes), need to map back to original indices
                unmapped_label = unmap_label(predicted_label, self.grade_offset)
            else:
                unmapped_label = predicted_label

            # Get all probabilities as dict
            all_probs = {}
            for model_label, prob in enumerate(sample_probs):
                if self.grade_offset > 0:
                    original_label = unmap_label(model_label, self.grade_offset)
                    grade = decode_grade(original_label)
                else:
                    grade = decode_grade(model_label)
                all_probs[grade] = float(prob)

            # Get top-k predictions
            top_k_indices = np.argsort(sample_probs)[-return_top_k:][::-1]
            top_k_predictions = []
            for idx in top_k_indices:
                if self.grade_offset > 0:
                    original_label = unmap_label(int(idx), self.grade_offset)
                    grade = decode_grade(original_label)
                else:
                    grade = decode_grade(int(idx))
                top_k_predictions.append((grade, float(sample_probs[idx])))

            results.append(
                {
                    "predicted_grade": decode_grade(unmapped_label),
                    "predicted_label": unmapped_label,
                    "confidence": confidence,
                    "all_probabilities": all_probs,
                    "top_k_predictions": top_k_predictions,
                }
            )

        # Return single dict if single sample, list otherwise
        return results[0] if len(results) == 1 else results

    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information:
                - 'model_type': Type of model (string)
                - 'num_parameters': Total number of parameters
                - 'num_classes': Number of grade classes
                - 'device': Device model is on
                - 'checkpoint_path': Path to loaded checkpoint

        Examples:
            >>> info = predictor.get_model_info()
            >>> print(f"Model type: {info['model_type']}")
            >>> print(f"Parameters: {info['num_parameters']:,}")
        """
        # Get model class name
        model_type = self.model.__class__.__name__

        # Count parameters
        num_parameters = sum(p.numel() for p in self.model.parameters())

        # Get number of classes - try common final layer names
        num_classes = None
        for attr_name in ["fc3", "fc", "classifier"]:
            if hasattr(self.model, attr_name):
                final_layer = getattr(self.model, attr_name)
                if isinstance(final_layer, torch.nn.Linear):
                    num_classes = final_layer.out_features
                    break

        # For Sequential models (FullyConnectedModel)
        if num_classes is None and hasattr(self.model, "network"):
            network = self.model.network
            if isinstance(network, torch.nn.Sequential):
                # Get last Linear layer in the sequence
                for layer in reversed(list(network)):
                    if isinstance(layer, torch.nn.Linear):
                        num_classes = layer.out_features
                        break

        # Fallback to default
        if num_classes is None:
            num_classes = get_num_grades()

        info = {
            "model_type": model_type,
            "num_parameters": num_parameters,
            "num_classes": num_classes,
            "device": self.device,
            "checkpoint_path": str(self.checkpoint_path),
        }

        # Add filtering information if model uses filtering
        if self.grade_offset > 0:
            from moonboard_core.grade_encoder import decode_grade

            info["filtered"] = True
            info["grade_offset"] = self.grade_offset
            info["min_grade"] = decode_grade(self.min_grade_index)
            info["max_grade"] = decode_grade(self.max_grade_index)
        else:
            info["filtered"] = False

        return info
