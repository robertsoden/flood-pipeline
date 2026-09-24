"""
Shared utilities and configuration for flood enrichment pipeline.
"""

# Import all configuration variables and functions
from .config import (
    # File paths
    PROJECT_ROOT,
    train_filepath,
    test_filepath,
    extraction_filepath,
    stage3_train_filepath,
    stage3_test_filepath,
    unlabeled_filepath,

    # Model configuration
    MODEL_CONFIG,

    # Stage configurations
    STAGE1_CONFIG,
    STAGE2_CONFIG,
    STAGE3_CONFIG,

    # Config helpers
    GLOBAL_DEFAULTS,
    get_config_value,
    get_temperature,
)

# Import all utility functions
from .utils import (
    prepare_data,
)

# Import logging utilities
from .logging_config import (
    setup_logger,
    log_section,
    log_config,
    log_stats,
)

# Import validation functions
from .validation import (
    validate_articles,
    validate_with_auto_detect,
    detect_article_format,
    generate_article_id,
    ensure_article_id,
    normalize_article_fields,
    ArticleBase,
    LabeledArticleNested,
    LabeledArticleFlat,
    UnlabeledArticle,
    Stage1Results,
    Stage2Results,
)

# Import checkpoint utilities
from .checkpoint import (
    CheckpointManager,
    filter_unprocessed,
)

# Import DSPy utilities
from .dspy_utils import (
    configure_dspy_lm,
    get_configured_lm,
    load_optimized_model,
    create_evaluator,
)

__all__ = [
    # Config - File paths
    'PROJECT_ROOT',
    'train_filepath',
    'test_filepath',
    'extraction_filepath',
    'stage3_train_filepath',
    'stage3_test_filepath',
    'unlabeled_filepath',

    # Config - Model settings
    'MODEL_CONFIG',

    # Config - Stage settings
    'STAGE1_CONFIG',
    'STAGE2_CONFIG',
    'STAGE3_CONFIG',

    # Config - Helpers
    'GLOBAL_DEFAULTS',
    'get_config_value',
    'get_temperature',

    # Utils - DSPy functions
    'prepare_data',

    # Logging - Functions
    'setup_logger',
    'log_section',
    'log_config',
    'log_stats',

    # Validation - Functions
    'validate_articles',
    'validate_with_auto_detect',
    'detect_article_format',
    'generate_article_id',
    'ensure_article_id',
    'normalize_article_fields',

    # Validation - Schemas
    'ArticleBase',
    'LabeledArticleNested',
    'LabeledArticleFlat',
    'UnlabeledArticle',
    'Stage1Results',
    'Stage2Results',

    # Checkpoint utilities
    'CheckpointManager',
    'filter_unprocessed',

    # DSPy utilities
    'configure_dspy_lm',
    'get_configured_lm',
    'load_optimized_model',
    'create_evaluator',
]