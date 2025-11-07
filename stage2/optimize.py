"""
Stage 2 Optimization: Train and optimize flood verification models
Run this once to create optimized models, then use process.py to apply them.
"""
import sys
from pathlib import Path
import json
import dspy
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f'stage2_optimize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# Import from shared config and stage2 modules
from shared.config import (
    train_filepath,
    test_filepath,
    MODEL_CONFIG,
    STAGE2_CONFIG,
    PROJECT_ROOT
)
from shared.utils import prepare_data
from stage2.signatures import floodIdentification, isOntario
from stage2.metrics import extraction_precision_focused_metric, ontario_correctness_metric

print("\n" + "="*70)
print("STAGE 2: MODEL OPTIMIZATION")
print("="*70)
print("\nThis script trains and optimizes the flood verification models.")
print("Run this once, then use process.py to apply the models.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  Model: {MODEL_CONFIG['name']}")
print(f"  Max bootstrapped demos: {STAGE2_CONFIG['max_bootstrapped_demos']}")
print(f"  Max labeled demos: {STAGE2_CONFIG['max_labeled_demos']}")
print(f"  Num candidate programs: {STAGE2_CONFIG['num_candidate_programs']}")
print(f"  Num threads: {STAGE2_CONFIG['num_threads']}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading labeled training/test data...")

# Load labeled data for optimization
with open(train_filepath, 'r') as file:
    train_set = prepare_data(json.load(file))
print(f"   Training examples: {len(train_set)}")

with open(test_filepath, 'r') as file:
    test_set = prepare_data(json.load(file))
print(f"   Test examples: {len(test_set)}")

# ============================================================================
# CONFIGURE DSPY
# ============================================================================

print("\n2. Configuring DSPy...")

# Configure language model
lm = dspy.LM(
    MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key'],
    temperature=MODEL_CONFIG.get('temperature', 1.0)  # Use config temperature or default to 1.0
)
dspy.configure(lm=lm)
print(f"   ✓ LM configured: {MODEL_CONFIG['name']}")

# ============================================================================
# OPTIMIZE FLOOD VERIFICATION MODEL
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZING FLOOD VERIFICATION MODEL")
print("="*70)
print("Goal: High-precision flood detection to reduce BERT false positives")

# Create predictor
flood_verifier = dspy.ChainOfThought(floodIdentification)

# Create evaluator
evaluate_flood = dspy.Evaluate(
    devset=test_set,
    metric=extraction_precision_focused_metric,
    num_threads=STAGE2_CONFIG['num_threads'],
    display_progress=True,
    display_table=True
)

# Baseline evaluation
print("\nEvaluating baseline flood verifier...")
baseline_flood = evaluate_flood(flood_verifier)
print(f"Baseline Score: {baseline_flood.score:.2f}%")

# Optimize
print("\nOptimizing flood verification...")
print(f"Using MIPROv2 optimizer for better prompt engineering.")
print("This may take a while...\n")

flood_optimizer = dspy.MIPROv2(
    metric=extraction_precision_focused_metric,
    auto="light",  # "light" = ~8 trials, faster optimization
    num_threads=STAGE2_CONFIG['num_threads'],
    verbose=True
)

optimized_flood_verifier = flood_optimizer.compile(
    flood_verifier,
    trainset=train_set,
    max_bootstrapped_demos=STAGE2_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE2_CONFIG['max_labeled_demos']
)

# Evaluate optimized
print("\nEvaluating optimized flood verifier...")
optimized_flood = evaluate_flood(optimized_flood_verifier)
print(f"Optimized Score: {optimized_flood.score:.2f}%")
print(f"Improvement: {optimized_flood.score - baseline_flood.score:.2f}%")

# Calculate detailed metrics
results = optimized_flood.results
tp = sum(1 for ex, pred, _ in results if ex.flood_mentioned and pred.flood_mentioned)
fp = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and pred.flood_mentioned)
fn = sum(1 for ex, pred, _ in results if ex.flood_mentioned and not pred.flood_mentioned)
tn = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and not pred.flood_mentioned)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nDetailed Metrics:")
print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")

# Save optimized model
model_path = MODELS_DIR / 'stage2_flood_verified.json'
optimized_flood_verifier.save(str(model_path))
print(f"\n✓ Flood verification model saved: {model_path}")

# ============================================================================
# OPTIMIZE ONTARIO FILTERING MODEL
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZING ONTARIO FILTERING MODEL")
print("="*70)
print("Goal: Identify floods that occurred in Ontario, Canada")

# Filter training/test data to flood-positive examples only
flood_positive_train = [ex for ex in train_set if ex.flood_mentioned]
flood_positive_test = [ex for ex in test_set if ex.flood_mentioned]

print(f"\nFlood-positive training examples: {len(flood_positive_train)}")
print(f"Flood-positive test examples: {len(flood_positive_test)}")

# Create predictor
ontario_checker = dspy.ChainOfThought(isOntario)

# Create evaluator
evaluate_ontario = dspy.Evaluate(
    devset=flood_positive_test,
    metric=ontario_correctness_metric,
    num_threads=STAGE2_CONFIG['num_threads'],
    display_progress=True,
    display_table=True
)

# Baseline evaluation
print("\nEvaluating baseline Ontario checker...")
baseline_ontario = evaluate_ontario(ontario_checker)
print(f"Baseline Score: {baseline_ontario.score:.2f}%")

# Optimize
print("\nOptimizing Ontario identification...")
print(f"Using MIPROv2 optimizer for better prompt engineering.")
print("This may take a while...\n")

ontario_optimizer = dspy.MIPROv2(
    metric=ontario_correctness_metric,
    auto="light",  # "light" = ~8 trials, faster optimization
    num_threads=STAGE2_CONFIG['num_threads'],
    verbose=True
)

optimized_ontario_checker = ontario_optimizer.compile(
    ontario_checker,
    trainset=flood_positive_train,
    max_bootstrapped_demos=STAGE2_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE2_CONFIG['max_labeled_demos']
)

# Evaluate optimized
print("\nEvaluating optimized Ontario checker...")
optimized_ontario = evaluate_ontario(optimized_ontario_checker)
print(f"Optimized Score: {optimized_ontario.score:.2f}%")
print(f"Improvement: {optimized_ontario.score - baseline_ontario.score:.2f}%")

# Calculate detailed metrics
results = optimized_ontario.results
tp = sum(1 for ex, pred, _ in results if ex.is_ontario and pred.is_ontario)
fp = sum(1 for ex, pred, _ in results if not ex.is_ontario and pred.is_ontario)
fn = sum(1 for ex, pred, _ in results if ex.is_ontario and not pred.is_ontario)
tn = sum(1 for ex, pred, _ in results if not ex.is_ontario and not pred.is_ontario)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nDetailed Metrics:")
print(f"  Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"  Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")

# Save optimized model
model_path = MODELS_DIR / 'stage2_ontario_filter.json'
optimized_ontario_checker.save(str(model_path))
print(f"\n✓ Ontario filtering model saved: {model_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)

print(f"\nOptimized models saved:")
print(f"  {MODELS_DIR / 'stage2_flood_verified.json'}")
print(f"  {MODELS_DIR / 'stage2_ontario_filter.json'}")

print(f"\nNext step:")
print(f"  Run: python stage2/process.py")
print(f"  This will apply the optimized models to all Stage 1 results.")

print("\n" + "="*70 + "\n")
