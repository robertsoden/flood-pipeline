"""
Stage 2 Optimization: Train and optimize flood verification models
"""

import weave
import sys
from pathlib import Path
import json
import dspy
import os
import logging
from datetime import datetime

# Weights & Measures (optional - disable with SKIP_WEAVE=1 for faster optimization):
if not os.getenv('SKIP_WEAVE'):
    weave.init('flood-stage2-optimization')
else:
    print("⚡ Weave tracing disabled for faster optimization")

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
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

# Capture all stdout to log file while still showing in terminal
class TeeLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.terminal = sys.stdout

    def write(self, message):
        if message.strip():  # Don't log empty lines
            self.logger.log(self.level, message.rstrip())
        self.terminal.write(message)

    def flush(self):
        self.terminal.flush()

sys.stdout = TeeLogger(logger, logging.INFO)

# Import from shared config and stage2 modules
from shared.config import (
    train_filepath,
    test_filepath,
    MODEL_CONFIG,
    STAGE2_CONFIG,
    PROJECT_ROOT,
    get_temperature
)
from shared.utils import prepare_data
from stage2.signatures import floodIdentification, isOntario
from stage2.metrics import extraction_precision_focused_metric, ontario_correctness_metric

print("\n" + "="*70)
print("STAGE 2: MODEL OPTIMIZATION (DIAGNOSTIC VERSION)")
print("="*70)
print("\nUsing ALL training data + detailed error analysis")
print("Expected time: 20-30 minutes\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Get temperature for optimization (allows exploration)
temperature = get_temperature(STAGE2_CONFIG, mode='optimization')

print(f"Configuration:")
print(f"  Model: {MODEL_CONFIG['name']}")
print(f"  Temperature: {temperature} (optimization mode)")
print(f"  Max bootstrapped demos: {STAGE2_CONFIG['max_bootstrapped_demos']}")
print(f"  Max labeled demos: {STAGE2_CONFIG['max_labeled_demos']}")
print(f"  Num threads: {STAGE2_CONFIG['num_threads']}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading labeled training/test data...")

# Load ALL labeled data (no sampling)
import random

with open(train_filepath, 'r') as file:
    full_train_set = prepare_data(json.load(file))

  # Sample N training examples (set to desired number: 100, 200, etc.)
TRAIN_SAMPLE_SIZE = STAGE2_CONFIG['train_sample_size']
if len(full_train_set) > TRAIN_SAMPLE_SIZE:
    random.seed(42)  # For reproducibility
    train_set = random.sample(full_train_set, TRAIN_SAMPLE_SIZE)
    print(f"   Training examples: {len(train_set)} (sampled from {len(full_train_set)})")
else:
    train_set = full_train_set
    print(f"   Training examples: {len(train_set)} (using ALL data)")

with open(test_filepath, 'r') as file:
    test_set = prepare_data(json.load(file))
    print(f"   Test examples: {len(test_set)}")

# Analyze training data composition
train_floods = sum(1 for ex in train_set if ex.flood_mentioned)
train_non_floods = len(train_set) - train_floods

print(f"\n   Training distribution:")
print(f"     Floods: {train_floods} ({train_floods/len(train_set):.1%})")
print(f"     Non-floods: {train_non_floods} ({train_non_floods/len(train_set):.1%})")

# Check for tricky non-flood patterns
print(f"\n   Analyzing non-flood training examples...")
metaphorical_count = 0
planning_count = 0
historical_count = 0

for ex in train_set:
    if not ex.flood_mentioned:
        text = ex.article_text.lower() if hasattr(ex, 'article_text') else ''
        
        if 'flood of' in text or 'flooded with' in text:
            metaphorical_count += 1
        if 'insurance' in text or 'planning' in text or 'prepare for' in text:
            planning_count += 1
        if 'anniversary' in text or 'commemorate' in text or 'years ago' in text:
            historical_count += 1

print(f"     Metaphorical patterns: {metaphorical_count}/{train_non_floods}")
print(f"     Planning/insurance: {planning_count}/{train_non_floods}")
print(f"     Historical mentions: {historical_count}/{train_non_floods}")

if metaphorical_count + planning_count + historical_count < 10:
    print(f"\n   ⚠️  WARNING: Few tricky non-flood examples!")
    print(f"   Model may struggle with metaphorical floods and planning articles.")
    print(f"   Consider adding more edge case examples to training data.")



# ============================================================================
# CONFIGURE DSPY
# ============================================================================

print("\n2. Configuring DSPy...")

# Configure language model with optimization temperature
temperature = get_temperature(STAGE2_CONFIG, mode='optimization')
lm = dspy.LM(
    MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key'],
    temperature=temperature
)
dspy.configure(lm=lm)
print(f"   ✓ LM configured: {MODEL_CONFIG['name']}")
print(f"   ✓ Temperature: {temperature} (optimization mode)")

# ============================================================================
# OPTIMIZE FLOOD VERIFICATION MODEL
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZING FLOOD VERIFICATION MODEL")
print("="*70)
print("Goal: High-precision flood detection to reduce BERT false positives")
print(f"Using all {len(train_set)} training examples\n")

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
print("Evaluating baseline flood verifier...")
baseline_flood = evaluate_flood(flood_verifier)
print(f"\n✓ Baseline Score: {baseline_flood.score:.2f}%")

# Calculate baseline metrics
results = baseline_flood.results
tp = sum(1 for ex, pred, _ in results if ex.flood_mentioned and pred.flood_mentioned)
fp = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and pred.flood_mentioned)
fn = sum(1 for ex, pred, _ in results if ex.flood_mentioned and not pred.flood_mentioned)
tn = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and not pred.flood_mentioned)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"Baseline Metrics:")
print(f"  Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# Optimize using BootstrapFewShotWithRandomSearch
print(f"\nOptimizing with BootstrapFewShotWithRandomSearch...")
print(f"Testing {STAGE2_CONFIG['num_candidate_programs']} candidate programs...")
print("This will take 45-90 minutes...\n")

from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Create progress-tracking wrapper for the metric
candidate_counter = {'count': 0, 'best_score': 0, 'scores': []}

def tracked_metric(example, prediction, trace=None):
    score = extraction_precision_focused_metric(example, prediction, trace)
    candidate_counter['count'] += 1
    candidate_counter['scores'].append(score)

    # Calculate stats every test set evaluation (160 examples)
    if candidate_counter['count'] % 160 == 0:
        candidate_num = candidate_counter['count'] // 160
        avg_score = sum(candidate_counter['scores'][-160:]) / 160

        if avg_score > candidate_counter['best_score']:
            candidate_counter['best_score'] = avg_score
            marker = " 🌟 NEW BEST"
        else:
            marker = ""

        print(f"  Candidate {candidate_num}/{STAGE2_CONFIG['num_candidate_programs']}: {avg_score:.1%}{marker}")

    return score

flood_optimizer = BootstrapFewShotWithRandomSearch(
    metric=tracked_metric,
    max_bootstrapped_demos=STAGE2_CONFIG['max_bootstrapped_demos'],
    max_labeled_demos=STAGE2_CONFIG['max_labeled_demos'],
    num_candidate_programs=STAGE2_CONFIG['num_candidate_programs'],
    num_threads=STAGE2_CONFIG['num_threads']
)

optimized_flood_verifier = flood_optimizer.compile(
    flood_verifier,
    trainset=train_set  # Use ALL training data
)

# Evaluate optimized
print("\nEvaluating optimized flood verifier...")
optimized_flood = evaluate_flood(optimized_flood_verifier)
print(f"\n✓ Optimized Score: {optimized_flood.score:.2f}%")
print(f"✓ Improvement: {optimized_flood.score - baseline_flood.score:+.2f}%")

# Calculate optimized metrics
results = optimized_flood.results
tp = sum(1 for ex, pred, _ in results if ex.flood_mentioned and pred.flood_mentioned)
fp = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and pred.flood_mentioned)
fn = sum(1 for ex, pred, _ in results if ex.flood_mentioned and not pred.flood_mentioned)
tn = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and not pred.flood_mentioned)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nOptimized Metrics:")
print(f"  Precision: {precision:.1%} | Recall: {recall:.1%} | F1: {f1:.1%}")
print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

# ============================================================================
# ERROR ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("DETAILED ERROR ANALYSIS")
print("="*70)

results = optimized_flood.results

# False positives (said flood, wasn't)
fps = [(ex, pred) for ex, pred, _ in results 
       if not ex.flood_mentioned and pred.flood_mentioned]

# False negatives (said not flood, was)
fns = [(ex, pred) for ex, pred, _ in results 
       if ex.flood_mentioned and not pred.flood_mentioned]

print(f"\n📊 Error Breakdown:")
print(f"   Total test examples: {len(results)}")
print(f"   Correct: {len(results)-len(fps)-len(fns)} ({(len(results)-len(fps)-len(fns))/len(results)*100:.1f}%)")
print(f"   False Positives: {len(fps)} ({len(fps)/len(results)*100:.1f}%)")
print(f"   False Negatives: {len(fns)} ({len(fns)/len(results)*100:.1f}%)")

# Analyze false positive patterns
print(f"\n🔴 FALSE POSITIVES ({len(fps)} total) - Model said FLOOD, actually NOT:")
print(f"   These are articles the model incorrectly classified as floods.\n")

fp_patterns = {'metaphorical': 0, 'planning': 0, 'historical': 0, 'other': 0}
for ex, pred in fps:
    text = ex.article_text.lower() if hasattr(ex, 'article_text') else ''
    if 'flood of' in text or 'flooded with' in text:
        fp_patterns['metaphorical'] += 1
    elif 'insurance' in text or 'planning' in text or 'prepare' in text:
        fp_patterns['planning'] += 1
    elif 'anniversary' in text or 'years ago' in text or 'commemorate' in text:
        fp_patterns['historical'] += 1
    else:
        fp_patterns['other'] += 1

print(f"   Pattern analysis:")
print(f"     Metaphorical floods: {fp_patterns['metaphorical']}/{len(fps)}")
print(f"     Planning/insurance: {fp_patterns['planning']}/{len(fps)}")
print(f"     Historical mentions: {fp_patterns['historical']}/{len(fps)}")
print(f"     Other: {fp_patterns['other']}/{len(fps)}")

print(f"\n   Sample false positives:")
for i, (ex, pred) in enumerate(fps[:5], 1):
    title = ex.title if hasattr(ex, 'title') else 'No title'
    text_preview = ex.article_text[:150] if hasattr(ex, 'article_text') else 'No text'
    reasoning = pred.reasoning if hasattr(pred, 'reasoning') else 'No reasoning'
    
    print(f"\n   {i}. Title: {title}")
    print(f"      Text: {text_preview}...")
    print(f"      Model said: {reasoning}")

# Analyze false negative patterns
print(f"\n🔵 FALSE NEGATIVES ({len(fns)} total) - Model said NOT FLOOD, actually WAS:")
print(f"   These are real floods the model missed.\n")

print(f"   Sample false negatives:")
for i, (ex, pred) in enumerate(fns[:5], 1):
    title = ex.title if hasattr(ex, 'title') else 'No title'
    text_preview = ex.article_text[:150] if hasattr(ex, 'article_text') else 'No text'
    reasoning = pred.reasoning if hasattr(pred, 'reasoning') else 'No reasoning'
    
    print(f"\n   {i}. Title: {title}")
    print(f"      Text: {text_preview}...")
    print(f"      Model said: {reasoning}")

# ============================================================================
# GPT-4O-MINI COMPARISON (OPTIONAL)
# ============================================================================

if os.getenv('OPENAI_API_KEY'):
    print("\n" + "="*70)
    print("GPT-4O-MINI COMPARISON TEST")
    print("="*70)
    print("Testing with GPT-4o-mini to determine if issue is model capability...\n")
    
    try:
        # Temporarily use GPT-4o-mini
        lm_gpt = dspy.LM(
            'openai/gpt-4o-mini',
            api_key=os.getenv('OPENAI_API_KEY'),
            temperature=temperature
        )
        dspy.configure(lm=lm_gpt)
        
        # Test baseline
        flood_verifier_gpt = dspy.ChainOfThought(floodIdentification)
        baseline_gpt = evaluate_flood(flood_verifier_gpt)
        
        print(f"GPT-4o-mini baseline: {baseline_gpt.score:.2f}%")
        print(f"Qwen baseline: {baseline_flood.score:.2f}%")
        print(f"Difference: {baseline_gpt.score - baseline_flood.score:+.2f}%")
        
        if baseline_gpt.score > baseline_flood.score + 10:
            print(f"\n⚠️  GPT-4o-mini significantly better!")
            print(f"   Your local model may not have sufficient capability.")
            print(f"   Consider using GPT-4o-mini for production.")
        elif baseline_gpt.score < baseline_flood.score + 5:
            print(f"\n✓ Models perform similarly.")
            print(f"  Issue is likely data/prompts, not model capability.")
        
        # Switch back to Qwen
        dspy.configure(lm=lm)
        
    except Exception as e:
        print(f"   ⚠️  GPT-4o-mini test failed: {e}")
        print(f"   Continuing with Qwen...")
        dspy.configure(lm=lm)
else:
    print("\n💡 TIP: Set OPENAI_API_KEY to compare with GPT-4o-mini")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("\n" + "="*70)
print("RECOMMENDATIONS")
print("="*70)

if precision >= 0.80:
    print(f"\n✅ Target precision (80%+) achieved!")
elif precision >= 0.75:
    print(f"\n⚠️  Close to target (75-80%).")
    print(f"\n   To improve:")
    
    if fp_patterns['metaphorical'] > len(fps) * 0.3:
        print(f"   • Add more metaphorical flood examples to training:")
        print(f"     - 'flood of applications'")
        print(f"     - 'flooded with calls'")
        print(f"     - 'flood of complaints'")
    
    if fp_patterns['planning'] > len(fps) * 0.3:
        print(f"   • Add more planning/insurance examples to training:")
        print(f"     - 'flood insurance rates'")
        print(f"     - 'preparing for flood season'")
        print(f"     - 'flood zone designation'")
    
    if fp_patterns['historical'] > len(fps) * 0.2:
        print(f"   • Add more historical reference examples to training:")
        print(f"     - 'anniversary of 1997 flood'")
        print(f"     - 'remembering the flood'")
        print(f"     - 'since the great flood of...'")
    
    if len(fns) > len(fps):
        print(f"   • Model is too conservative (missing real floods)")
        print(f"     - Review false negatives above")
        print(f"     - Add similar flood examples to training")
        print(f"     - Make signature more permissive")
else:
    print(f"\n❌ Below 75% - significant improvements needed.")
    print(f"\n   Priority actions:")
    print(f"   1. Review false positives/negatives above")
    print(f"   2. Add 20-30 examples similar to the errors")
    print(f"   3. Clarify signature based on error patterns")
    print(f"   4. Consider trying GPT-4o-mini (set OPENAI_API_KEY)")

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

# Filter training/test data to flood-positive examples only
flood_positive_train = [ex for ex in train_set if ex.flood_mentioned]
flood_positive_test = [ex for ex in test_set if ex.flood_mentioned]

print(f"\nFlood-positive training examples: {len(flood_positive_train)}")
print(f"Flood-positive test examples: {len(flood_positive_test)}")

if len(flood_positive_train) < 10:
    print("\n⚠️  WARNING: Very few flood-positive training examples!")
    print("   Ontario filtering may not be well-optimized.")

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
print(f"✓ Baseline Score: {baseline_ontario.score:.2f}%")

# Optimize
print(f"\nOptimizing with BootstrapFewShotWithRandomSearch...")
print(f"Testing {STAGE2_CONFIG['num_candidate_programs']} candidate programs...")

# Create progress-tracking wrapper for Ontario metric
ontario_counter = {'count': 0, 'best_score': 0, 'scores': []}
ontario_test_size = len(flood_positive_test)

def tracked_ontario_metric(example, prediction, trace=None):
    score = ontario_correctness_metric(example, prediction, trace)
    ontario_counter['count'] += 1
    ontario_counter['scores'].append(score)

    # Calculate stats every test set evaluation
    if ontario_counter['count'] % ontario_test_size == 0:
        candidate_num = ontario_counter['count'] // ontario_test_size
        avg_score = sum(ontario_counter['scores'][-ontario_test_size:]) / ontario_test_size

        if avg_score > ontario_counter['best_score']:
            ontario_counter['best_score'] = avg_score
            marker = " 🌟 NEW BEST"
        else:
            marker = ""

        print(f"  Candidate {candidate_num}/{STAGE2_CONFIG['num_candidate_programs']}: {avg_score:.1%}{marker}")

    return score

ontario_optimizer = BootstrapFewShotWithRandomSearch(
    metric=tracked_ontario_metric,
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
print(f"\n✓ Optimized Score: {optimized_ontario.score:.2f}%")
print(f"✓ Improvement: {optimized_ontario.score - baseline_ontario.score:+.2f}%")

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

print(f"\n📊 Final Performance:")
print(f"  Flood Detection: {optimized_flood.score:.1f}% accuracy")
print(f"  Ontario Filtering: {optimized_ontario.score:.1f}% accuracy")

print(f"\n📁 Optimized models saved:")
print(f"  {MODELS_DIR / 'stage2_flood_verified.json'}")
print(f"  {MODELS_DIR / 'stage2_ontario_filter.json'}")

print(f"\n✅ Next step:")
print(f"  Review the error analysis above, then run:")
print(f"  python stage2/process.py")

print("\n" + "="*70 + "\n")