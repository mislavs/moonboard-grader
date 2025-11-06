"""
Command Registry

Central registry for all CLI commands and their argument parsers.
"""

from .train import setup_train_parser
from .evaluate import setup_evaluate_parser
from .predict import setup_predict_parser


def setup_parsers(parser):
    """
    Setup all command parsers.
    
    Creates subparsers for train, evaluate, and predict commands
    with their respective arguments.
    
    Args:
        parser: Main ArgumentParser instance
        
    Returns:
        Configured subparsers object
    """
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    subparsers.required = True
    
    # Register all commands
    setup_train_parser(subparsers)
    setup_evaluate_parser(subparsers)
    setup_predict_parser(subparsers)
    
    return subparsers

