"""
Predict Command

Handles making predictions on new climbing problems using trained models.
"""

import sys
import json
from pathlib import Path
import torch

from .utils import print_section_header, print_completion_message
from src import Predictor
from src.grade_encoder import encode_grade


def setup_predict_parser(subparsers):
    """
    Setup argument parser for predict command.
    
    Args:
        subparsers: ArgumentParser subparsers object
        
    Returns:
        Configured predict parser
    """
    predict_parser = subparsers.add_parser('predict', help='Make predictions on new problems')
    predict_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )
    predict_parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input problem JSON file'
    )
    predict_parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU usage (default: use CUDA if available)'
    )
    predict_parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Return top K predictions (default: 3)'
    )
    predict_parser.add_argument(
        '--output',
        type=str,
        help='Save predictions to JSON file'
    )
    predict_parser.set_defaults(func=predict_command)
    return predict_parser


def predict_command(args):
    """
    Execute prediction command.
    
    Args:
        args: Parsed command-line arguments
    """
    print_section_header("MOONBOARD GRADE PREDICTION - PREDICTION")
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    print(f"\n✓ Loading model from: {checkpoint_path}")
    
    # Load predictor
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"✓ Using device: {device}")
    
    predictor = Predictor(str(checkpoint_path), device=device)
    model_info = predictor.get_model_info()
    
    print(f"\n🧠 Model: {model_info['model_type']} ({model_info['num_parameters']:,} parameters)")
    
    # Load input problem
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    print(f"\n📂 Loading problem from: {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Handle both single problem and problem with 'data' array
    if 'data' in data and isinstance(data['data'], list):
        problems = data['data']
        batch_mode = True
    elif 'moves' in data:
        problems = [data]
        batch_mode = False
    else:
        print("❌ Error: Invalid input format. Expected 'moves' field or 'data' array.")
        sys.exit(1)
    
    print(f"   Found {len(problems)} problem(s)")
    
    # Make predictions
    top_k = args.top_k if args.top_k else 3
    
    if batch_mode:
        print(f"\n🔮 Making predictions (top-{top_k})...")
        predictions = predictor.predict_batch(problems, return_top_k=top_k)
        
        for i, pred in enumerate(predictions):
            if 'error' in pred:
                print(f"\n❌ Problem {i+1}: {pred['error']}")
            else:
                print(f"\n📊 Problem {i+1}:")
                print(f"   Predicted Grade: {pred['predicted_grade']}")
                print(f"   Confidence: {pred['confidence']*100:.2f}%")
                print(f"\n   Top {top_k} Predictions:")
                for j, (grade, prob) in enumerate(pred['top_k_predictions'], 1):
                    print(f"      {j}. {grade:<4} ({prob*100:.2f}%)")
    else:
        print(f"\n🔮 Making prediction (top-{top_k})...")
        pred = predictor.predict(problems[0], return_top_k=top_k)
        
        print(f"\n📊 Results:")
        print(f"   Predicted Grade: {pred['predicted_grade']}")
        print(f"   Confidence: {pred['confidence']*100:.2f}%")
        print(f"\n   Top {top_k} Predictions:")
        for j, (grade, prob) in enumerate(pred['top_k_predictions'], 1):
            print(f"      {j}. {grade:<4} ({prob*100:.2f}%)")
        
        # Show actual grade if present
        if 'grade' in problems[0]:
            actual_grade = problems[0]['grade']
            print(f"\n   Actual Grade: {actual_grade}")
            
            # Check if prediction was correct
            if pred['predicted_grade'] == actual_grade:
                print(f"   ✅ Exact match!")
            else:
                # Check if within tolerance
                pred_idx = encode_grade(pred['predicted_grade'])
                actual_idx = encode_grade(actual_grade)
                diff = abs(pred_idx - actual_idx)
                
                if diff == 1:
                    print(f"   ⚠️  Off by 1 grade")
                elif diff == 2:
                    print(f"   ⚠️  Off by 2 grades")
                else:
                    print(f"   ❌ Off by {diff} grades")
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            if batch_mode:
                json.dump(predictions, f, indent=2)
            else:
                json.dump(pred, f, indent=2)
        
        print(f"\n✓ Saved predictions to: {output_path}")
    
    print_completion_message("✅ Prediction completed successfully!")

