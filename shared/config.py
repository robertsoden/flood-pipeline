from pathlib import Path
from typing import Any, Optional

# Project root is parent of shared/ directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data sources
# Stage 1 (BERT) uses original splits
bert_train_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_train_70pct.json'
bert_test_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_test_30pct.json'
extraction_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'extraction_reserve_100.json'

# Stage 2 (DSPy) uses complete labeled data with titles
train_filepath = PROJECT_ROOT / 'stage2' / 'data' / 'stage2_train_70pct.json'
test_filepath = PROJECT_ROOT / 'stage2' / 'data' / 'stage2_test_30pct.json'

# Unlabeled data for inference
unlabeled_filepath = PROJECT_ROOT / 'data' / 'articles_restructured.json'


# Global defaults for magic numbers used across stages
GLOBAL_DEFAULTS = {
    'progress_interval': 100,
    'batch_progress_interval': 4000,
    'pseudo_label_confidence': 0.95,
    'threshold_min': 0.10,
    'threshold_max': 0.50,
    'threshold_step': 0.01,
    'temperature_optimization': 0.7,  # Higher for exploration during training
    'temperature_inference': 0.1,      # Lower for deterministic predictions
}


# Model configuration
MODEL_CONFIG = {
    'name': 'ollama_chat/qwen2.5:14b-instruct-q5_K_M',
    'api_base': 'http://localhost:11434',
    'api_key': '',
}

# Optimization settings
STAGE1_CONFIG = {
      'bert_model': 'distilbert-base-uncased',
      'bert_model_dir': PROJECT_ROOT / 'models' / 'balanced_high_recall_iter0',
      'batch_size': 16,
      'learning_rate': 2e-5,
      'num_epochs': 4,
      'max_length': 512,
      'threshold': 0.5,
      'target_recall': 0.95,
      # Stage 1 specific magic numbers
      'pseudo_label_confidence': 0.95,
      'max_pseudo_labels_per_iteration': 3000,
      'num_iterations': 3,
      'recall_weight_multiplier': 1.1,
}

STAGE2_CONFIG = {
    'max_bootstrapped_demos': 3,       # Reduced for faster optimization
    'max_labeled_demos': 3,            # Reduced for faster optimization
    'num_candidate_programs': 10,      # Reduced for faster optimization
    'num_threads': 8,
    'temperature': 'temperature_optimization',  # Use optimization temperature
}

STAGE3_CONFIG = {
    'max_bootstrapped_demos': 3,       # Few-shot examples for optimization
    'max_labeled_demos': 3,            # Labeled examples for optimization
    'num_threads': 8,
}


def get_config_value(key: str, stage_config: Optional[dict] = None) -> Any:
    """
    Get configuration value with stage-specific override support.

    Priority:
    1. Stage-specific config (if provided and key exists)
    2. Global defaults
    3. None if not found

    Args:
        key: Configuration key to lookup
        stage_config: Optional stage-specific config dict (e.g., STAGE1_CONFIG)

    Returns:
        Configuration value or None if not found

    Example:
        >>> interval = get_config_value('progress_interval', STAGE1_CONFIG)
        >>> confidence = get_config_value('pseudo_label_confidence', STAGE1_CONFIG)
    """
    if stage_config and key in stage_config:
        value = stage_config[key]
        # If value is a string reference to another config key, resolve it
        if isinstance(value, str) and value in GLOBAL_DEFAULTS:
            return GLOBAL_DEFAULTS[value]
        return value
    return GLOBAL_DEFAULTS.get(key)


def get_temperature(stage_config: Optional[dict] = None, mode: str = 'optimization') -> float:
    """
    Get temperature for LLM calls with proper optimization vs inference distinction.

    Args:
        stage_config: Optional stage-specific config dict
        mode: Either 'optimization' or 'inference'

    Returns:
        Temperature value (float)

    Example:
        >>> temp = get_temperature(STAGE2_CONFIG, mode='optimization')  # 0.7
        >>> temp = get_temperature(STAGE2_CONFIG, mode='inference')     # 0.1
    """
    key = f'temperature_{mode}'
    return get_config_value(key, stage_config)