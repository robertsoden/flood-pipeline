"""
Stage 3 Optimization: Train location and date extraction model
"""
import weave
import sys
from pathlib import Path
import json
import dspy
import os
import logging
from datetime import datetime

# Weave tracing
if not os.getenv('SKIP_WEAVE'):
    weave.init('flood-stage3-optimization')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f'stage3_optimize_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from shared.config import MODEL_CONFIG, PROJECT_ROOT, STAGE3_CONFIG, get_temperature
from shared.utils import prepare_data
from stage3.signatures import FloodLocationExtraction
from stage3.metrics import location_extraction_metric

print("\n" + "="*70)
print("STAGE 3: LOCATION/DATE EXTRACTION OPTIMIZATION")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

train_file = PROJECT_ROOT / 'stage3' / 'data' / 'stage3_train_70pct.json'
test_file = PROJECT_ROOT / 'stage3' / 'data' / 'stage3_test_30pct.json'

print("\n1. Loading training/test data...")

with open(train_file, 'r') as f:
    train_set = prepare_data(json.load(f))

with open(test_file, 'r') as f:
    test_set = prepare_data(json.load(f))

print(f"   Training examples: {len(train_set)}")
print(f"   Test examples: {len(test_set)}")

# ============================================================================
# CONFIGURE DSPY
# ============================================================================

print("\n2. Configuring DSPy...")

temperature = get_temperature(STAGE3_CONFIG, mode='optimization')

# Configure LiteLLM
import litellm
litellm.num_retries = 5
litellm.request_timeout = 120

# Check for Anthropic API key
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
if anthropic_key:
    model_name = 'anthropic/claude-sonnet-4-20250514'
    lm = dspy.LM(
        model_name,
        api_key=anthropic_key,
        temperature=temperature,
        num_retries=5
    )
    print(f"   ✓ Using Claude Sonnet 4 for optimization")
else:
    model_name = MODEL_CONFIG['name']
    lm_kwargs = {'temperature': temperature}
    if MODEL_CONFIG.get('api_base'):
        lm_kwargs['api_base'] = MODEL_CONFIG['api_base']
    if MODEL_CONFIG.get('api_key'):
        lm_kwargs['api_key'] = MODEL_CONFIG['api_key']
    lm = dspy.LM(model_name, **lm_kwargs)
    print(f"   ✓ Using configured model: {model_name}")

dspy.configure(lm=lm)

# ============================================================================
# OPTIMIZE EXTRACTION MODEL
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZING LOCATION/DATE EXTRACTION")
print("="*70)

# Create predictor
extractor = dspy.ChainOfThought(FloodLocationExtraction)

# Evaluate baseline
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=location_extraction_metric,
    num_threads=STAGE3_CONFIG.get('num_threads', 1),
    display_progress=True,
    display_table=True,
    max_errors=10,
    failure_score=0.0
)

print("Evaluating baseline...")
baseline = evaluate(extractor)
print(f"\n✓ Baseline Score: {baseline.score:.2f}%")

# Optimize
print(f"\nOptimizing with BootstrapFewShotWithRandomSearch...")

from dspy.teleprompt import BootstrapFewShotWithRandomSearch

optimizer = BootstrapFewShotWithRandomSearch(
    metric=location_extraction_metric,
    max_bootstrapped_demos=STAGE3_CONFIG.get('max_bootstrapped_demos', 3),
    max_labeled_demos=STAGE3_CONFIG.get('max_labeled_demos', 3),
    num_candidate_programs=STAGE3_CONFIG.get('num_candidate_programs', 10),
    num_threads=STAGE3_CONFIG.get('num_threads', 1)
)

optimized_extractor = optimizer.compile(extractor, trainset=train_set)

# Evaluate optimized
print("\nEvaluating optimized extractor...")
optimized = evaluate(optimized_extractor)
print(f"\n✓ Optimized Score: {optimized.score:.2f}%")
print(f"✓ Improvement: {optimized.score - baseline.score:+.2f}%")

# Save model
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(exist_ok=True)
model_path = MODELS_DIR / 'stage3_location_extraction.json'
optimized_extractor.save(str(model_path))
print(f"\n✓ Model saved: {model_path}")

print("\n" + "="*70)
print("OPTIMIZATION COMPLETE")
print("="*70)
print(f"\nNext step: python stage3/process.py")
