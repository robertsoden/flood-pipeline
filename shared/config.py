from pathlib import Path

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


# Model configuration
MODEL_CONFIG = {
    'name': 'ollama_chat/qwen2.5:14b-instruct-q5_K_M',
    'api_base': 'http://localhost:11434',
    'api_key': '',
    'temperature': 0.7   # Higher temperature for better optimization exploration
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
      'target_recall': 0.95
}

STAGE2_CONFIG = {
    'max_bootstrapped_demos': 3,       # Reduced for faster optimization
    'max_labeled_demos': 3,            # Reduced for faster optimization
    'num_candidate_programs': 10,      # Reduced for faster optimization
    'num_threads': 8
}

STAGE3_CONFIG = {
    'max_bootstrapped_demos': 3,       # Few-shot examples for optimization
    'max_labeled_demos': 3,            # Labeled examples for optimization
    'num_threads': 8,
    'temperature_optimization': 0.7,   # Higher temperature for optimization exploration
    'temperature_inference': 0.1       # Lower temperature for deterministic inference
}