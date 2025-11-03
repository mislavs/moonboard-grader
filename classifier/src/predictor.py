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

from .grade_encoder import decode_grade, get_num_grades
from .grid_builder import create_grid_tensor
from .models import FullyConnectedModel, ConvolutionalModel
from .advanced_models import ResidualCNN


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
    
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: str = 'cpu'
    ):
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
        if device not in ['cpu', 'cuda']:
            raise ValueError(f"Invalid device: {device}. Must be 'cpu' or 'cuda'")
        
        # Check CUDA availability
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError("CUDA device requested but CUDA is not available")
        
        self.device = device
        self.checkpoint_path = Path(checkpoint_path)
        
        # Check if checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load the checkpoint
        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Validate checkpoint structure
        if 'model_state_dict' not in checkpoint:
            raise ValueError("Invalid checkpoint: missing 'model_state_dict'")
        
        # Infer model architecture from state dict
        state_dict = checkpoint['model_state_dict']
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
        # Check for advanced model (has residual blocks or attention)
        if any(key.startswith('res') for key in state_dict.keys()) or \
           any(key.startswith('att') for key in state_dict.keys()):
            # Residual CNN with residual blocks and attention
            # fc3 is the final Linear layer
            if 'fc3.weight' in state_dict:
                num_classes = state_dict['fc3.weight'].shape[0]
            else:
                num_classes = get_num_grades()
            return ResidualCNN(num_classes=num_classes)
        # Get number of classes from final layer
        elif 'network.7.weight' in state_dict:
            # Fully Connected Model (Sequential container)
            # network.7 is the final Linear layer
            num_classes = state_dict['network.7.weight'].shape[0]
            return FullyConnectedModel(num_classes=num_classes)
        elif 'fc2.weight' in state_dict:
            # Convolutional Model
            # fc2 is the final Linear layer
            num_classes = state_dict['fc2.weight'].shape[0]
            return ConvolutionalModel(num_classes=num_classes)
        else:
            raise ValueError(
                "Cannot infer model architecture from state dict. "
                "Unknown model type."
            )
    
    def predict(
        self,
        problem: Dict,
        return_top_k: int = 3
    ) -> Dict:
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
        if 'moves' not in problem:
            raise ValueError("Problem must contain 'moves' key")
        
        # Convert problem to tensor
        try:
            tensor = create_grid_tensor(problem['moves'])
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
        
        # Get all probabilities as dict
        all_probs = {}
        for label, prob in enumerate(probs_array):
            grade = decode_grade(label)
            all_probs[grade] = float(prob)
        
        # Get top-k predictions
        top_k_indices = np.argsort(probs_array)[-return_top_k:][::-1]
        top_k_predictions = [
            (decode_grade(int(idx)), float(probs_array[idx]))
            for idx in top_k_indices
        ]
        
        return {
            'predicted_grade': decode_grade(predicted_label),
            'predicted_label': predicted_label,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'top_k_predictions': top_k_predictions
        }
    
    def predict_batch(
        self,
        problems: List[Dict],
        return_top_k: int = 3
    ) -> List[Dict]:
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
        
        # Process each problem individually
        # Note: We could optimize this by batching, but individual processing
        # is more robust to varying problem sizes and easier to debug
        results = []
        for i, problem in enumerate(problems):
            try:
                result = self.predict(problem, return_top_k=return_top_k)
                results.append(result)
            except Exception as e:
                raise ValueError(f"Failed to process problem {i}: {e}")
        
        return results
    
    def predict_from_tensor(
        self,
        tensor: Union[np.ndarray, torch.Tensor],
        return_top_k: int = 3
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
        
        for sample_probs in probs_array:
            # Get predicted class and confidence
            predicted_label = int(np.argmax(sample_probs))
            confidence = float(sample_probs[predicted_label])
            
            # Get all probabilities as dict
            all_probs = {}
            for label, prob in enumerate(sample_probs):
                grade = decode_grade(label)
                all_probs[grade] = float(prob)
            
            # Get top-k predictions
            top_k_indices = np.argsort(sample_probs)[-return_top_k:][::-1]
            top_k_predictions = [
                (decode_grade(int(idx)), float(sample_probs[idx]))
                for idx in top_k_indices
            ]
            
            results.append({
                'predicted_grade': decode_grade(predicted_label),
                'predicted_label': predicted_label,
                'confidence': confidence,
                'all_probabilities': all_probs,
                'top_k_predictions': top_k_predictions
            })
        
        # Return single dict if single sample, list otherwise
        return results[0] if len(results) == 1 else results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information:
                - 'model_type': Type of model (FC or CNN)
                - 'num_parameters': Total number of parameters
                - 'num_classes': Number of grade classes
                - 'device': Device model is on
                - 'checkpoint_path': Path to loaded checkpoint
                
        Examples:
            >>> info = predictor.get_model_info()
            >>> print(f"Model type: {info['model_type']}")
            >>> print(f"Parameters: {info['num_parameters']:,}")
        """
        # Determine model type
        if isinstance(self.model, FullyConnectedModel):
            model_type = 'FullyConnected'
        elif isinstance(self.model, ConvolutionalModel):
            model_type = 'Convolutional'
        else:
            model_type = 'Unknown'
        
        # Count parameters
        num_parameters = sum(p.numel() for p in self.model.parameters())
        
        # Get number of classes from model
        if isinstance(self.model, FullyConnectedModel):
            # Fully connected model - final layer is network[7]
            num_classes = self.model.network[7].out_features
        elif isinstance(self.model, ConvolutionalModel):
            # Convolutional model - final layer is fc2
            num_classes = self.model.fc2.out_features
        else:
            num_classes = get_num_grades()
        
        return {
            'model_type': model_type,
            'num_parameters': num_parameters,
            'num_classes': num_classes,
            'device': self.device,
            'checkpoint_path': str(self.checkpoint_path)
        }

