"""
Shared utilities and configuration for flood enrichment pipeline.
"""

# Import all configuration variables and functions
from .config import (
    # File paths
    train_filepath,
    val_filepath,
    test_filepath,
    extraction_filepath,
    unlabeled_filepath,
    
    # Model configuration
    MODEL_CONFIG,
    
    # Stage configurations
    STAGE1_CONFIG,
    STAGE2_CONFIG,
)

# Import all utility functions
from .utils import (
    prepare_data,
)

__all__ = [
    # Config - File paths
    'train_filepath',
    'val_filepath',
    'test_filepath',
    'extraction_filepath',
    'unlabeled_filepath',
    
    # Config - Model settings
    'MODEL_CONFIG',
    
    # Config - Stage settings
    'STAGE1_CONFIG',
    'STAGE2_CONFIG',
    
    # Utils - DSPy functions
    'prepare_data',
]