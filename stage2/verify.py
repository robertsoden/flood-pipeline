"""
Stage 2: Flood Verification and Ontario Filtering
High-precision LLM-based verification of BERT-filtered articles
"""
import sys
from pathlib import Path
import json
import dspy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
print("STAGE 2: FLOOD VERIFICATION AND ONTARIO FILTERING")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input/output paths
STAGE1_RESULTS = PROJECT_ROOT / 'results' / 'predicted_floods.json'
OUTPUT_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Stage 1 input: {STAGE1_RESULTS}")
print(f"  Model: {MODEL_CONFIG['name']}")
print(f"  Max bootstrapped demos: {STAGE2_CONFIG['max_bootstrapped_demos']}")
print(f"  Max labeled demos: {STAGE2_CONFIG['max_labeled_demos']}")
print(f"  Num candidate programs: {STAGE2_CONFIG['num_candidate_programs']}")

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

# Load Stage 1 BERT results
print("\n2. Loading Stage 1 BERT results...")
try:
    with open(STAGE1_RESULTS, 'r') as file:
        stage1_articles = json.load(file)
    print(f"   Loaded {len(stage1_articles):,} articles from Stage 1")
except FileNotFoundError:
    print(f"   ERROR: Stage 1 results not found at {STAGE1_RESULTS}")
    print(f"   Please run stage1-bert/bert-inference.py first")
    sys.exit(1)

# ============================================================================
# CONFIGURE DSPY
# ============================================================================

print("\n3. Configuring DSPy...")

# Configure language model
lm = dspy.LM(
    MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key']
)
dspy.configure(lm=lm)
print(f"   ✓ LM configured: {MODEL_CONFIG['name']}")

# ============================================================================
# STAGE 2.1: FLOOD VERIFICATION (HIGH PRECISION)
# ============================================================================

print("\n" + "="*70)
print("STAGE 2.1: FLOOD VERIFICATION (HIGH PRECISION)")
print("="*70)
print("Goal: Re-verify flood mentions with LLM to reduce false positives")

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
flood_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=extraction_precision_focused_metric,
    max_bootstrapped_demos=STAGE2_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE2_CONFIG['max_labeled_demos'],
    num_candidate_programs=STAGE2_CONFIG['num_candidate_programs'],
    num_threads=STAGE2_CONFIG['num_threads']
)

optimized_flood_verifier = flood_optimizer.compile(
    flood_verifier,
    trainset=train_set
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
print(f"\n✓ Model saved: {model_path}")

# ============================================================================
# APPLY FLOOD VERIFICATION TO STAGE 1 RESULTS
# ============================================================================

print("\n" + "="*70)
print("APPLYING FLOOD VERIFICATION TO STAGE 1 RESULTS")
print("="*70)

verified_floods = []
flood_verification_stats = {'verified': 0, 'rejected': 0}

print(f"Processing {len(stage1_articles):,} articles...")

for i, article in enumerate(stage1_articles):
    # Create DSPy example
    example_input = dspy.Example(
        article_text=article.get('full_text', '')
    ).with_inputs('article_text')

    # Predict
    prediction = optimized_flood_verifier(**example_input.inputs())

    # Add Stage 2 results to article
    article['stage2'] = {
        'flood_verified': prediction.flood_mentioned,
        'flood_reasoning': prediction.reasoning,
        'confidence': article.get('confidence', 'UNKNOWN')
    }

    # Track verified floods
    if prediction.flood_mentioned:
        verified_floods.append(article)
        flood_verification_stats['verified'] += 1
    else:
        flood_verification_stats['rejected'] += 1

    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1:,}/{len(stage1_articles):,} articles...")

print(f"\n✓ Flood verification complete!")
print(f"  Verified floods: {flood_verification_stats['verified']:,} ({flood_verification_stats['verified']/len(stage1_articles):.1%})")
print(f"  Rejected: {flood_verification_stats['rejected']:,} ({flood_verification_stats['rejected']/len(stage1_articles):.1%})")

# Save verified floods
verified_floods_path = OUTPUT_DIR / 'stage2_verified_floods.json'
with open(verified_floods_path, 'w') as f:
    json.dump(verified_floods, f, indent=2)
print(f"✓ Verified floods saved: {verified_floods_path}")

# ============================================================================
# STAGE 2.2: ONTARIO FILTERING
# ============================================================================

print("\n" + "="*70)
print("STAGE 2.2: ONTARIO FILTERING")
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
ontario_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=ontario_correctness_metric,
    max_bootstrapped_demos=STAGE2_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE2_CONFIG['max_labeled_demos'],
    num_candidate_programs=STAGE2_CONFIG['num_candidate_programs'],
    num_threads=STAGE2_CONFIG['num_threads']
)

optimized_ontario_checker = ontario_optimizer.compile(
    ontario_checker,
    trainset=flood_positive_train
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
print(f"\n✓ Model saved: {model_path}")

# ============================================================================
# APPLY ONTARIO FILTERING TO VERIFIED FLOODS
# ============================================================================

print("\n" + "="*70)
print("APPLYING ONTARIO FILTERING TO VERIFIED FLOODS")
print("="*70)

ontario_floods = []
ontario_stats = {'ontario': 0, 'non_ontario': 0}

print(f"Processing {len(verified_floods):,} verified floods...")

for i, article in enumerate(verified_floods):
    # Create DSPy example
    example_input = dspy.Example(
        article_text=article.get('full_text', '')
    ).with_inputs('article_text')

    # Predict
    prediction = optimized_ontario_checker(**example_input.inputs())

    # Add Ontario results to Stage 2 data
    article['stage2']['is_ontario'] = prediction.is_ontario
    article['stage2']['ontario_reasoning'] = prediction.reasoning

    # Track Ontario floods
    if prediction.is_ontario:
        ontario_floods.append(article)
        ontario_stats['ontario'] += 1
    else:
        ontario_stats['non_ontario'] += 1

    # Progress indicator
    if (i + 1) % 100 == 0:
        print(f"  Processed {i+1:,}/{len(verified_floods):,} articles...")

print(f"\n✓ Ontario filtering complete!")
print(f"  Ontario floods: {ontario_stats['ontario']:,} ({ontario_stats['ontario']/len(verified_floods):.1%})")
print(f"  Non-Ontario: {ontario_stats['non_ontario']:,} ({ontario_stats['non_ontario']/len(verified_floods):.1%})")

# Save Ontario floods
ontario_floods_path = OUTPUT_DIR / 'stage2_ontario_floods.json'
with open(ontario_floods_path, 'w') as f:
    json.dump(ontario_floods, f, indent=2)
print(f"✓ Ontario floods saved: {ontario_floods_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("STAGE 2 COMPLETE")
print("="*70)

print(f"\nPipeline Summary:")
print(f"  Stage 1 input (BERT): {len(stage1_articles):,} articles")
print(f"  After flood verification: {len(verified_floods):,} articles ({len(verified_floods)/len(stage1_articles):.1%})")
print(f"  After Ontario filtering: {len(ontario_floods):,} articles ({len(ontario_floods)/len(stage1_articles):.1%})")

print(f"\nOutput Files:")
print(f"  Verified floods: {verified_floods_path}")
print(f"  Ontario floods: {ontario_floods_path}")
print(f"  Models: {MODELS_DIR / 'stage2_flood_verified.json'}")
print(f"          {MODELS_DIR / 'stage2_ontario_filter.json'}")

print("\n" + "="*70)
print("Ready for Stage 3: Location/Date Extraction")
print("="*70 + "\n")
