"""
Predictor Module

Inference interface for predicting grades from beta solver output.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from moonboard_core import decode_grade, get_all_grades

from .dataset import FeatureNormalizer, extract_features
from .model import TransformerSequenceClassifier


class Predictor:
    """
    Inference wrapper for trained grade classifier.
    
    Loads a trained model and normalizer, provides methods for
    predicting grades from beta solver move sequences.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        normalizer_path: str,
        device: str = 'cpu'
    ):
        """
        Load model and normalizer for inference.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            normalizer_path: Path to normalizer (.npz file)
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.device = device
        
        # Load normalizer
        self.normalizer = FeatureNormalizer.load(normalizer_path)
        
        # Load model from checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        model_config = checkpoint['model_config']
        self.model = TransformerSequenceClassifier(**model_config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.grade_names = get_all_grades()
    
    def predict(self, problem: Dict) -> Dict:
        """
        Predict grade from a single beta solver problem.
        
        Args:
            problem: Dict containing 'moves' key with list of move dicts
            
        Returns:
            Dict with:
                - predicted_grade: String grade (e.g., "7A+")
                - predicted_index: Integer grade index
                - confidence: Probability of predicted grade
                - all_probabilities: Dict mapping grade names to probabilities
        """
        if 'moves' not in problem or len(problem['moves']) == 0:
            raise ValueError("Problem must contain non-empty 'moves' list")
        
        # Extract and normalize features
        features = extract_features(problem['moves'])
        normalized = self.normalizer.transform([features])[0]
        
        # Create tensors
        x = torch.FloatTensor(normalized).unsqueeze(0).to(self.device)
        mask = torch.ones(1, len(normalized)).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(x, mask)
            probs = F.softmax(logits, dim=1)
        
        pred_idx = probs.argmax().item()
        confidence = probs[0, pred_idx].item()
        
        # Build probability dict for all grades
        all_probs = {
            self.grade_names[i]: probs[0, i].item()
            for i in range(len(self.grade_names))
        }
        
        return {
            'predicted_grade': decode_grade(pred_idx),
            'predicted_index': pred_idx,
            'confidence': confidence,
            'all_probabilities': all_probs
        }
    
    def predict_batch(self, problems: List[Dict]) -> List[Dict]:
        """
        Predict grades for multiple problems.
        
        Args:
            problems: List of problem dicts, each containing 'moves'
            
        Returns:
            List of prediction dicts (same format as predict())
        """
        return [self.predict(p) for p in problems]
    
    def predict_with_alternatives(
        self,
        problem: Dict,
        top_k: int = 3
    ) -> Dict:
        """
        Predict grade with top-k alternatives.
        
        Args:
            problem: Dict containing 'moves'
            top_k: Number of top predictions to return
            
        Returns:
            Dict with predicted_grade, confidence, and alternatives list
        """
        result = self.predict(problem)
        
        # Sort probabilities
        sorted_probs = sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        alternatives = [
            {'grade': grade, 'probability': prob}
            for grade, prob in sorted_probs[:top_k]
        ]
        
        return {
            'predicted_grade': result['predicted_grade'],
            'confidence': result['confidence'],
            'alternatives': alternatives
        }
    
    def compare_with_actual(
        self,
        problem: Dict,
        actual_grade: Optional[str] = None
    ) -> Dict:
        """
        Predict and compare with actual grade if provided.
        
        Args:
            problem: Dict containing 'moves' and optionally 'grade'
            actual_grade: Override actual grade (uses problem['grade'] if not provided)
            
        Returns:
            Dict with prediction and comparison info
        """
        result = self.predict(problem)
        
        actual = actual_grade or problem.get('grade')
        
        if actual:
            from moonboard_core import encode_grade
            actual_idx = encode_grade(actual)
            pred_idx = result['predicted_index']
            error = pred_idx - actual_idx
            
            result['actual_grade'] = actual
            result['actual_index'] = actual_idx
            result['error'] = error
            result['correct'] = error == 0
            result['within_1'] = abs(error) <= 1
            result['within_2'] = abs(error) <= 2
        
        return result

