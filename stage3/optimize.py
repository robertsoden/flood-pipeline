"""
Stage 3 Optimization: Train and optimize location/date extraction models
Run this once to create optimized models, then use process.py to apply them.
"""
import sys
from pathlib import Path
import json
import dspy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared modules
from shared.config import (
    MODEL_CONFIG,
    STAGE3_CONFIG,
    PROJECT_ROOT,
    get_temperature
)
from shared.utils import prepare_data
from shared.logging_config import setup_logger, log_section, log_config, log_stats
from stage3.signatures import LocationExtraction, DateExtraction
from stage3.metrics import (
    location_extraction_metric,
    date_extraction_metric,
    combined_extraction_metric
)

# Setup logging
logger = setup_logger(__name__, 'stage3_optimize', PROJECT_ROOT)

log_section(logger, "STAGE 3: MODEL OPTIMIZATION")
logger.info("Training models to extract flood location and date from articles.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

STAGE3_DATA_DIR = PROJECT_ROOT / 'stage3' / 'data'
train_filepath = STAGE3_DATA_DIR / 'stage3_train_70pct.json'
test_filepath = STAGE3_DATA_DIR / 'stage3_test_30pct.json'

config_display = {
    'Model': MODEL_CONFIG['name'],
    'Max bootstrapped demos': STAGE3_CONFIG['max_bootstrapped_demos'],
    'Max labeled demos': STAGE3_CONFIG['max_labeled_demos'],
    'Num threads': STAGE3_CONFIG['num_threads'],
}
log_config(logger, config_display, "Configuration")

# ============================================================================
# LOAD DATA
# ============================================================================

logger.info("\n1. Loading labeled training/test data...")

# Load labeled Ontario flood data
try:
    with open(train_filepath, 'r') as file:
        train_set = prepare_data(json.load(file))
    logger.info(f"   Training examples: {len(train_set)}")
except FileNotFoundError:
    logger.error(f"   ❌ ERROR: Training data not found at {train_filepath}")
    logger.error(f"   Please run: python stage3/create_splits.py first")
    sys.exit(1)

try:
    with open(test_filepath, 'r') as file:
        test_set = prepare_data(json.load(file))
    logger.info(f"   Test examples: {len(test_set)}")
except FileNotFoundError:
    logger.error(f"   ❌ ERROR: Test data not found at {test_filepath}")
    logger.error(f"   Please run: python stage3/create_splits.py first")
    sys.exit(1)

# ============================================================================
# CONFIGURE DSPY
# ============================================================================

logger.info("\n2. Configuring DSPy...")

# Configure language model with optimization temperature
temperature = get_temperature(STAGE3_CONFIG, mode='optimization')
lm = dspy.LM(
    MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key'],
    temperature=temperature
)
dspy.configure(lm=lm)
logger.info(f"   ✓ LM configured: {MODEL_CONFIG['name']}")
logger.info(f"   ✓ Temperature: {temperature} (optimization mode)")

# ============================================================================
# OPTIMIZE LOCATION EXTRACTION MODEL
# ============================================================================

log_section(logger, "OPTIMIZING LOCATION EXTRACTION MODEL")
logger.info("Goal: Extract city, town, region, or province where flood occurred")

# Create predictor
location_extractor = dspy.ChainOfThought(LocationExtraction)

# Create evaluator
evaluate_location = dspy.Evaluate(
    devset=test_set,
    metric=location_extraction_metric,
    num_threads=STAGE3_CONFIG['num_threads'],
    display_progress=True,
    display_table=True
)

# Baseline evaluation
logger.info("\nEvaluating baseline location extractor...")
baseline_location = evaluate_location(location_extractor)
logger.info(f"Baseline Score: {baseline_location.score:.2f}%")

# Optimize
logger.info("\nOptimizing location extraction...")
logger.info("Using MIPROv2 optimizer for better prompt engineering.")
logger.info("This may take a while...\n")

location_optimizer = dspy.MIPROv2(
    metric=location_extraction_metric,
    auto="light",  # "light" = ~8 trials, faster optimization
    num_threads=STAGE3_CONFIG['num_threads'],
    verbose=True
)

optimized_location_extractor = location_optimizer.compile(
    location_extractor,
    trainset=train_set,
    max_bootstrapped_demos=STAGE3_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE3_CONFIG['max_labeled_demos']
)

# Evaluate optimized
logger.info("\nEvaluating optimized location extractor...")
optimized_location = evaluate_location(optimized_location_extractor)
logger.info(f"Optimized Score: {optimized_location.score:.2f}%")
logger.info(f"Improvement: {optimized_location.score - baseline_location.score:.2f}%")

# Calculate accuracy
results = optimized_location.results
correct = sum(1 for ex, pred, score in results if score > 0.0)
partial = sum(1 for ex, pred, score in results if 0.0 < score < 1.0)
wrong = sum(1 for ex, pred, score in results if score == 0.0)

accuracy = correct / len(results) if results else 0.0

stats = {
    'Total': len(results),
    'Correct': f"{correct} ({correct/len(results):.1%})",
    'Partial': f"{partial} ({partial/len(results):.1%})",
    'Wrong': f"{wrong} ({wrong/len(results):.1%})",
}
log_stats(logger, stats, "Detailed Metrics")

# Save optimized model
model_path = MODELS_DIR / 'stage3_location_extractor.json'
optimized_location_extractor.save(str(model_path))
logger.info(f"\n✓ Location extraction model saved: {model_path}")

# ============================================================================
# OPTIMIZE DATE EXTRACTION MODEL
# ============================================================================

log_section(logger, "OPTIMIZING DATE EXTRACTION MODEL")
logger.info("Goal: Extract when flood occurred (month and year)")

# Create predictor
date_extractor = dspy.ChainOfThought(DateExtraction)

# Create evaluator
evaluate_date = dspy.Evaluate(
    devset=test_set,
    metric=date_extraction_metric,
    num_threads=STAGE3_CONFIG['num_threads'],
    display_progress=True,
    display_table=True
)

# Baseline evaluation
logger.info("\nEvaluating baseline date extractor...")
baseline_date = evaluate_date(date_extractor)
logger.info(f"Baseline Score: {baseline_date.score:.2f}%")

# Optimize
logger.info("\nOptimizing date extraction...")
logger.info("Using MIPROv2 optimizer for better prompt engineering.")
logger.info("This may take a while...\n")

date_optimizer = dspy.MIPROv2(
    metric=date_extraction_metric,
    auto="light",  # "light" = ~8 trials, faster optimization
    num_threads=STAGE3_CONFIG['num_threads'],
    verbose=True
)

optimized_date_extractor = date_optimizer.compile(
    date_extractor,
    trainset=train_set,
    max_bootstrapped_demos=STAGE3_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE3_CONFIG['max_labeled_demos']
)

# Evaluate optimized
logger.info("\nEvaluating optimized date extractor...")
optimized_date = evaluate_date(optimized_date_extractor)
logger.info(f"Optimized Score: {optimized_date.score:.2f}%")
logger.info(f"Improvement: {optimized_date.score - baseline_date.score:.2f}%")

# Calculate accuracy
results = optimized_date.results
correct = sum(1 for ex, pred, score in results if score > 0.0)
partial = sum(1 for ex, pred, score in results if 0.0 < score < 1.0)
wrong = sum(1 for ex, pred, score in results if score == 0.0)

stats = {
    'Total': len(results),
    'Correct': f"{correct} ({correct/len(results):.1%})",
    'Partial': f"{partial} ({partial/len(results):.1%})",
    'Wrong': f"{wrong} ({wrong/len(results):.1%})",
}
log_stats(logger, stats, "Detailed Metrics")

# Save optimized model
model_path = MODELS_DIR / 'stage3_date_extractor.json'
optimized_date_extractor.save(str(model_path))
logger.info(f"\n✓ Date extraction model saved: {model_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log_section(logger, "OPTIMIZATION COMPLETE")

logger.info(f"\nOptimized models saved:")
logger.info(f"  {MODELS_DIR / 'stage3_location_extractor.json'}")
logger.info(f"  {MODELS_DIR / 'stage3_date_extractor.json'}")

logger.info(f"\nNext step:")
logger.info(f"  Run: python stage3/process.py")
logger.info(f"  This will apply the optimized models to Stage 2 results.")

logger.info("\n" + "="*70 + "\n")
