"""
Stage 2 Processing: Apply optimized models to Stage 1 results
Run this after optimize.py to process all BERT-filtered articles.
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
log_file = LOGS_DIR / f'stage2_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

# Import from shared config
from shared.config import MODEL_CONFIG, PROJECT_ROOT, STAGE2_CONFIG, get_temperature, get_config_value

print("\n" + "="*70)
print("STAGE 2: PROCESSING ARTICLES")
print("="*70)
print("\nApplying optimized models to Stage 1 BERT results.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

STAGE1_RESULTS = PROJECT_ROOT / 'results' / 'predicted_floods.json'
OUTPUT_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

FLOOD_MODEL_PATH = MODELS_DIR / 'stage2_flood_verified.json'
ONTARIO_MODEL_PATH = MODELS_DIR / 'stage2_ontario_filter.json'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  Stage 1 input: {STAGE1_RESULTS}")
print(f"  Flood model: {FLOOD_MODEL_PATH}")
print(f"  Ontario model: {ONTARIO_MODEL_PATH}")
print(f"  LLM: {MODEL_CONFIG['name']}")

# ============================================================================
# CHECK FOR OPTIMIZED MODELS
# ============================================================================

print("\n1. Checking for optimized models...")

if not FLOOD_MODEL_PATH.exists():
    print(f"   ❌ ERROR: Flood verification model not found!")
    print(f"   Please run: python stage2/optimize.py first")
    sys.exit(1)

if not ONTARIO_MODEL_PATH.exists():
    print(f"   ❌ ERROR: Ontario filtering model not found!")
    print(f"   Please run: python stage2/optimize.py first")
    sys.exit(1)

print(f"   ✓ Found flood verification model")
print(f"   ✓ Found Ontario filtering model")

# ============================================================================
# LOAD STAGE 1 RESULTS
# ============================================================================

print("\n2. Loading Stage 1 BERT results...")

try:
    with open(STAGE1_RESULTS, 'r') as file:
        stage1_articles = json.load(file)
    print(f"   ✓ Loaded {len(stage1_articles):,} articles from Stage 1")
except FileNotFoundError:
    print(f"   ❌ ERROR: Stage 1 results not found at {STAGE1_RESULTS}")
    print(f"   Please run stage1-bert/bert-inference.py first")
    sys.exit(1)

# ============================================================================
# CONFIGURE DSPY AND LOAD MODELS
# ============================================================================

print("\n3. Loading optimized models...")

# Configure language model with inference temperature for deterministic predictions
temperature = get_temperature(STAGE2_CONFIG, mode='inference')
lm = dspy.LM(
    MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key'],
    temperature=temperature
)
dspy.configure(lm=lm)
print(f"   ✓ LM configured with temperature: {temperature} (inference mode)")

# Load optimized models
from stage2.signatures import floodIdentification, isOntario

flood_verifier = dspy.ChainOfThought(floodIdentification)
flood_verifier.load(str(FLOOD_MODEL_PATH))
print(f"   ✓ Loaded flood verification model")

ontario_checker = dspy.ChainOfThought(isOntario)
ontario_checker.load(str(ONTARIO_MODEL_PATH))
print(f"   ✓ Loaded Ontario filtering model")

# ============================================================================
# APPLY FLOOD VERIFICATION
# ============================================================================

print("\n" + "="*70)
print("STEP 1: FLOOD VERIFICATION")
print("="*70)
print(f"Processing {len(stage1_articles):,} articles...")

progress_interval = get_config_value('progress_interval', STAGE2_CONFIG)
verified_floods = []
flood_verification_stats = {'verified': 0, 'rejected': 0}

for i, article in enumerate(stage1_articles):
    # Create DSPy example
    example_input = dspy.Example(
        title=article.get('title', ''),
        article_text=article.get('full_text', '')
    ).with_inputs('title', 'article_text')

    # Predict
    try:
        prediction = flood_verifier(**example_input.inputs())

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
    except Exception as e:
        print(f"   ⚠ Error processing article {i}: {e}")
        # Keep article but mark as unprocessed
        article['stage2'] = {
            'flood_verified': False,
            'flood_reasoning': f"Error: {str(e)}",
            'confidence': 'ERROR'
        }
        flood_verification_stats['rejected'] += 1

    # Progress indicator
    if (i + 1) % progress_interval == 0:
        print(f"  Processed {i+1:,}/{len(stage1_articles):,} articles... "
              f"({flood_verification_stats['verified']} verified, "
              f"{flood_verification_stats['rejected']} rejected)")

print(f"\n✓ Flood verification complete!")
print(f"  Verified floods: {flood_verification_stats['verified']:,} ({flood_verification_stats['verified']/len(stage1_articles):.1%})")
print(f"  Rejected: {flood_verification_stats['rejected']:,} ({flood_verification_stats['rejected']/len(stage1_articles):.1%})")

# Save verified floods
verified_floods_path = OUTPUT_DIR / 'stage2_verified_floods.json'
with open(verified_floods_path, 'w') as f:
    json.dump(verified_floods, f, indent=2)
print(f"✓ Verified floods saved: {verified_floods_path}")

# ============================================================================
# APPLY ONTARIO FILTERING
# ============================================================================

print("\n" + "="*70)
print("STEP 2: ONTARIO FILTERING")
print("="*70)
print(f"Processing {len(verified_floods):,} verified floods...")

ontario_floods = []
ontario_stats = {'ontario': 0, 'non_ontario': 0}

for i, article in enumerate(verified_floods):
    # Create DSPy example
    example_input = dspy.Example(
        title=article.get('title', ''),
        article_text=article.get('full_text', '')
    ).with_inputs('title', 'article_text')

    # Predict
    try:
        prediction = ontario_checker(**example_input.inputs())

        # Add Ontario results to Stage 2 data
        article['stage2']['is_ontario'] = prediction.is_ontario
        article['stage2']['ontario_reasoning'] = prediction.reasoning

        # Track Ontario floods
        if prediction.is_ontario:
            ontario_floods.append(article)
            ontario_stats['ontario'] += 1
        else:
            ontario_stats['non_ontario'] += 1
    except Exception as e:
        print(f"   ⚠ Error processing article {i}: {e}")
        # Mark as non-Ontario on error
        article['stage2']['is_ontario'] = False
        article['stage2']['ontario_reasoning'] = f"Error: {str(e)}"
        ontario_stats['non_ontario'] += 1

    # Progress indicator
    if (i + 1) % progress_interval == 0:
        print(f"  Processed {i+1:,}/{len(verified_floods):,} articles... "
              f"({ontario_stats['ontario']} Ontario, "
              f"{ontario_stats['non_ontario']} non-Ontario)")

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

print("\n" + "="*70)
print("Ready for Stage 3: Location/Date Extraction")
print("="*70 + "\n")
