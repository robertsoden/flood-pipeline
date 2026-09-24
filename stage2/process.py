"""
Stage 2 Processing: Apply optimized models to Stage 1 results
Run this after optimize.py to process all BERT-filtered articles.

Features:
- Checkpoint/resume support for long-running processing
- Article ID normalization for traceability
- Input validation
- Shared logging configuration
"""
import sys
from pathlib import Path
import json
import dspy
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared utilities
from shared import (
    MODEL_CONFIG, PROJECT_ROOT, STAGE2_CONFIG,
    get_temperature, get_config_value,
    setup_logger, log_section, log_config,
    CheckpointManager, filter_unprocessed,
    configure_dspy_lm, load_optimized_model,
    ensure_article_id, normalize_article_fields,
)

# Setup logging using shared config
logger = setup_logger(__name__, 'stage2_process', PROJECT_ROOT)

# Argument parsing
parser = argparse.ArgumentParser(description='Stage 2: LLM flood verification')
parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
parser.add_argument('--clear-checkpoint', action='store_true', help='Clear existing checkpoint and start fresh')
parser.add_argument('--sample', type=int, default=None, help='Process only first N articles')
parser.add_argument('--checkpoint-interval', type=int, default=100, help='Save checkpoint every N articles')
args = parser.parse_args()

log_section(logger, "STAGE 2: PROCESSING ARTICLES")
print("\nApplying optimized models to Stage 1 BERT results.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

STAGE1_RESULTS = PROJECT_ROOT / 'results' / 'predicted_floods.json'
OUTPUT_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

FLOOD_MODEL_PATH = MODELS_DIR / 'stage2_flood_verified.json'
ONTARIO_MODEL_PATH = MODELS_DIR / 'stage2_ontario_filter.json'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

config_display = {
    'Stage 1 input': str(STAGE1_RESULTS),
    'Flood model': str(FLOOD_MODEL_PATH),
    'Ontario model': str(ONTARIO_MODEL_PATH),
    'LLM': MODEL_CONFIG['name'],
    'Resume mode': args.resume,
    'Checkpoint interval': args.checkpoint_interval,
}
log_config(logger, config_display)

# ============================================================================
# CHECK FOR OPTIMIZED MODELS
# ============================================================================

print("\n1. Checking for optimized models...")

if not FLOOD_MODEL_PATH.exists():
    logger.error(f"Flood verification model not found at {FLOOD_MODEL_PATH}")
    print(f"   Please run: python stage2/optimize.py first")
    sys.exit(1)

if not ONTARIO_MODEL_PATH.exists():
    logger.error(f"Ontario filtering model not found at {ONTARIO_MODEL_PATH}")
    print(f"   Please run: python stage2/optimize.py first")
    sys.exit(1)

logger.info("Found flood verification model")
logger.info("Found Ontario filtering model")

# ============================================================================
# LOAD STAGE 1 RESULTS
# ============================================================================

print("\n2. Loading Stage 1 BERT results...")

try:
    with open(STAGE1_RESULTS, 'r') as file:
        stage1_articles = json.load(file)

    # Normalize article IDs for all articles
    for article in stage1_articles:
        normalize_article_fields(article)

    logger.info(f"Loaded {len(stage1_articles):,} articles from Stage 1")

    # Apply sample limit if specified
    if args.sample:
        stage1_articles = stage1_articles[:args.sample]
        logger.info(f"Processing sample of {len(stage1_articles)} articles")

except FileNotFoundError:
    logger.error(f"Stage 1 results not found at {STAGE1_RESULTS}")
    print(f"   Please run stage1-bert/bert-inference.py first")
    sys.exit(1)

# ============================================================================
# INITIALIZE CHECKPOINTS
# ============================================================================

print("\n3. Initializing checkpoint system...")

flood_checkpoint = CheckpointManager(
    'stage2_flood',
    CHECKPOINT_DIR,
    save_interval=args.checkpoint_interval
)

ontario_checkpoint = CheckpointManager(
    'stage2_ontario',
    CHECKPOINT_DIR,
    save_interval=args.checkpoint_interval
)

# Handle checkpoint modes
if args.clear_checkpoint:
    flood_checkpoint.clear()
    ontario_checkpoint.clear()
    logger.info("Cleared existing checkpoints")
elif args.resume:
    flood_loaded = flood_checkpoint.load()
    ontario_loaded = ontario_checkpoint.load()
    if flood_loaded:
        logger.info(f"Resuming flood verification: {len(flood_checkpoint.processed_ids)} already processed")
    if ontario_loaded:
        logger.info(f"Resuming Ontario filtering: {len(ontario_checkpoint.processed_ids)} already processed")

# ============================================================================
# CONFIGURE DSPY AND LOAD MODELS
# ============================================================================

print("\n4. Loading optimized models...")

# Configure language model using shared utility
temperature = get_temperature(STAGE2_CONFIG, mode='inference')
configure_dspy_lm(MODEL_CONFIG, temperature=temperature, mode='inference')

# Load optimized models using shared utility
from stage2.signatures import floodIdentification, isOntario

flood_verifier = load_optimized_model(FLOOD_MODEL_PATH, floodIdentification)
logger.info("Loaded flood verification model")

ontario_checker = load_optimized_model(ONTARIO_MODEL_PATH, isOntario)
logger.info("Loaded Ontario filtering model")

# ============================================================================
# APPLY FLOOD VERIFICATION
# ============================================================================

log_section(logger, "STEP 1: FLOOD VERIFICATION")

# Filter to unprocessed articles if resuming
articles_to_process = stage1_articles
if args.resume and flood_checkpoint.processed_ids:
    articles_to_process = filter_unprocessed(stage1_articles, flood_checkpoint)
    # Get already verified floods from checkpoint
    verified_floods = flood_checkpoint.get_results()
else:
    verified_floods = []

print(f"Processing {len(articles_to_process):,} articles...")
if len(articles_to_process) < len(stage1_articles):
    print(f"  (Skipping {len(stage1_articles) - len(articles_to_process)} already processed)")

progress_interval = get_config_value('progress_interval', STAGE2_CONFIG)
flood_verification_stats = {'verified': len(verified_floods), 'rejected': 0, 'errors': 0}

for i, article in enumerate(articles_to_process):
    article_id = article.get('article_id')

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
            flood_checkpoint.mark_processed(article_id, article)
            flood_verification_stats['verified'] += 1
        else:
            flood_checkpoint.mark_processed(article_id)
            flood_verification_stats['rejected'] += 1

    except Exception as e:
        logger.warning(f"Error processing article {article_id}: {e}")
        article['stage2'] = {
            'flood_verified': False,
            'flood_reasoning': f"Error: {str(e)}",
            'confidence': 'ERROR'
        }
        flood_checkpoint.mark_processed(article_id)
        flood_verification_stats['rejected'] += 1
        flood_verification_stats['errors'] += 1

    # Checkpoint save
    if flood_checkpoint.should_save():
        flood_checkpoint.save()

    # Progress indicator
    if (i + 1) % progress_interval == 0:
        print(f"  Processed {i+1:,}/{len(articles_to_process):,} articles... "
              f"({flood_verification_stats['verified']} verified, "
              f"{flood_verification_stats['rejected']} rejected)")

# Final checkpoint save
flood_checkpoint.save(force=True)

total_processed = len(stage1_articles)
logger.info(f"Flood verification complete: {flood_verification_stats['verified']} verified, "
            f"{flood_verification_stats['rejected']} rejected, {flood_verification_stats['errors']} errors")

# Save verified floods
verified_floods_path = OUTPUT_DIR / 'stage2_verified_floods.json'
with open(verified_floods_path, 'w') as f:
    json.dump(verified_floods, f, indent=2)
logger.info(f"Verified floods saved: {verified_floods_path}")

# ============================================================================
# APPLY ONTARIO FILTERING
# ============================================================================

log_section(logger, "STEP 2: ONTARIO FILTERING")

# Filter to unprocessed articles if resuming
floods_to_process = verified_floods
if args.resume and ontario_checkpoint.processed_ids:
    floods_to_process = filter_unprocessed(verified_floods, ontario_checkpoint)
    ontario_floods = ontario_checkpoint.get_results()
else:
    ontario_floods = []

print(f"Processing {len(floods_to_process):,} verified floods...")
if len(floods_to_process) < len(verified_floods):
    print(f"  (Skipping {len(verified_floods) - len(floods_to_process)} already processed)")

ontario_stats = {'ontario': len(ontario_floods), 'non_ontario': 0, 'errors': 0}

for i, article in enumerate(floods_to_process):
    article_id = article.get('article_id')

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
            ontario_checkpoint.mark_processed(article_id, article)
            ontario_stats['ontario'] += 1
        else:
            ontario_checkpoint.mark_processed(article_id)
            ontario_stats['non_ontario'] += 1

    except Exception as e:
        logger.warning(f"Error processing article {article_id}: {e}")
        article['stage2']['is_ontario'] = False
        article['stage2']['ontario_reasoning'] = f"Error: {str(e)}"
        ontario_checkpoint.mark_processed(article_id)
        ontario_stats['non_ontario'] += 1
        ontario_stats['errors'] += 1

    # Checkpoint save
    if ontario_checkpoint.should_save():
        ontario_checkpoint.save()

    # Progress indicator
    if (i + 1) % progress_interval == 0:
        print(f"  Processed {i+1:,}/{len(floods_to_process):,} articles... "
              f"({ontario_stats['ontario']} Ontario, "
              f"{ontario_stats['non_ontario']} non-Ontario)")

# Final checkpoint save
ontario_checkpoint.save(force=True)

if len(verified_floods) > 0:
    logger.info(f"Ontario filtering complete: {ontario_stats['ontario']} Ontario, "
                f"{ontario_stats['non_ontario']} non-Ontario, {ontario_stats['errors']} errors")

# Save Ontario floods
ontario_floods_path = OUTPUT_DIR / 'stage2_ontario_floods.json'
with open(ontario_floods_path, 'w') as f:
    json.dump(ontario_floods, f, indent=2)
logger.info(f"Ontario floods saved: {ontario_floods_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log_section(logger, "STAGE 2 COMPLETE")

total_input = len(stage1_articles)
print(f"\nPipeline Summary:")
print(f"  Stage 1 input (BERT): {total_input:,} articles")
print(f"  After flood verification: {len(verified_floods):,} articles ({len(verified_floods)/max(total_input,1):.1%})")
print(f"  After Ontario filtering: {len(ontario_floods):,} articles ({len(ontario_floods)/max(total_input,1):.1%})")

print(f"\nOutput Files:")
print(f"  Verified floods: {verified_floods_path}")
print(f"  Ontario floods: {ontario_floods_path}")
print(f"  Checkpoints: {CHECKPOINT_DIR}")

logger.info(f"Stage 2 complete: {len(ontario_floods)} Ontario flood articles ready for Stage 3")

print("\n" + "="*70)
print("Ready for Stage 3: Location/Date Extraction")
print("  Run: python stage3/process_ner.py")
print("="*70 + "\n")
