from pathlib import Path

# Project root is parent of shared/ directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data sources
train_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_train_486.json'
val_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_val_65.json'
test_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_test_66.json'
extraction_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'extraction_reserve_100.json'
unlabeled_filepath = PROJECT_ROOT / 'data' / 'articles_restructured.json'


# Model configuration
MODEL_CONFIG = {
    'name': 'ollama_chat/qwen2.5:14b-instruct-q5_K_M',
    'api_base': 'http://localhost:11434',
    'api_key': ''
}

# Optimization settings
STAGE1_CONFIG = {    
      'bert_model': 'distilbert-base-uncased',
      'bert_model_dir': PROJECT_ROOT / 'models' / 'bert_flood_classifier',
      'batch_size': 16,
      'learning_rate': 2e-5,
      'num_epochs': 4,
      'max_length': 512,
      'threshold': 0.5,
      'target_recall': 0.95,
}

STAGE2_CONFIG = {
    'max_bootstrapped_demos': 8,
    'max_labeled_demos': 8,
    'num_candidate_programs': 30,
    'num_threads': 1
}