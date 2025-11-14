"""
Stage 3 Processing: Apply optimized extraction models to Stage 2 results
Run this after optimize.py to extract location and date from all verified Ontario floods.
"""
import sys
from pathlib import Path
import json
import dspy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared config
from shared.config import (
    MODEL_CONFIG,
    PROJECT_ROOT,
    STAGE3_CONFIG,
    get_temperature,
    get_config_value
)
from shared.logging_config import setup_logger, log_section, log_config

# Setup logging
logger = setup_logger(__name__, 'stage3_process', PROJECT_ROOT)

log_section(logger, "STAGE 3: PROCESSING ARTICLES")
logger.info("Applying optimized extraction models to Stage 2 results.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

STAGE2_RESULTS = PROJECT_ROOT / 'results' / 'stage2_ontario_floods.json'
OUTPUT_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

LOCATION_MODEL_PATH = MODELS_DIR / 'stage3_location_extractor.json'
DATE_MODEL_PATH = MODELS_DIR / 'stage3_date_extractor.json'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config_display = {
    'Stage 2 input': str(STAGE2_RESULTS),
    'Location model': str(LOCATION_MODEL_PATH),
    'Date model': str(DATE_MODEL_PATH),
    'LLM': MODEL_CONFIG['name'],
}
log_config(logger, config_display, "Configuration")

# ============================================================================
# CHECK FOR OPTIMIZED MODELS
# ============================================================================

logger.info("\n1. Checking for optimized models...")

if not LOCATION_MODEL_PATH.exists():
    logger.error(f"   ❌ ERROR: Location extraction model not found!")
    logger.error(f"   Please run: python stage3/optimize.py first")
    sys.exit(1)

if not DATE_MODEL_PATH.exists():
    logger.error(f"   ❌ ERROR: Date extraction model not found!")
    logger.error(f"   Please run: python stage3/optimize.py first")
    sys.exit(1)

logger.info(f"   ✓ Found location extraction model")
logger.info(f"   ✓ Found date extraction model")

# ============================================================================
# LOAD STAGE 2 RESULTS
# ============================================================================

logger.info("\n2. Loading Stage 2 Ontario flood results...")

try:
    with open(STAGE2_RESULTS, 'r') as file:
        stage2_articles = json.load(file)
    logger.info(f"   ✓ Loaded {len(stage2_articles):,} Ontario flood articles from Stage 2")
except FileNotFoundError:
    logger.error(f"   ❌ ERROR: Stage 2 results not found at {STAGE2_RESULTS}")
    logger.error(f"   Please run stage2/process.py first")
    sys.exit(1)

# ============================================================================
# CONFIGURE DSPY AND LOAD MODELS
# ============================================================================

logger.info("\n3. Loading optimized models...")

# Configure language model with inference temperature for deterministic predictions
temperature = get_temperature(STAGE3_CONFIG, mode='inference')
lm = dspy.LM(
    MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key'],
    temperature=temperature
)
dspy.configure(lm=lm)
logger.info(f"   ✓ LM configured with temperature: {temperature} (inference mode)")

# Load optimized models
from stage3.signatures import LocationExtraction, DateExtraction

location_extractor = dspy.ChainOfThought(LocationExtraction)
location_extractor.load(str(LOCATION_MODEL_PATH))
logger.info(f"   ✓ Loaded location extraction model")

date_extractor = dspy.ChainOfThought(DateExtraction)
date_extractor.load(str(DATE_MODEL_PATH))
logger.info(f"   ✓ Loaded date extraction model")

# ============================================================================
# EXTRACT LOCATION AND DATE
# ============================================================================

log_section(logger, "EXTRACTING LOCATION AND DATE")
logger.info(f"Processing {len(stage2_articles):,} articles...\n")

progress_interval = get_config_value('progress_interval', STAGE3_CONFIG)
extraction_stats = {
    'total': len(stage2_articles),
    'location_extracted': 0,
    'date_extracted': 0,
    'both_extracted': 0,
    'errors': 0
}

for i, article in enumerate(stage2_articles):
    # Create DSPy example
    example_input = dspy.Example(
        title=article.get('title', ''),
        article_text=article.get('full_text', ''),
        publication_date=article.get('publication_date', '')
    ).with_inputs('title', 'article_text', 'publication_date')

    # Initialize stage3 results
    article['stage3'] = {}

    # Extract location
    try:
        location_pred = location_extractor(**example_input.inputs())
        article['stage3']['location'] = location_pred.location
        article['stage3']['location_reasoning'] = location_pred.reasoning

        if location_pred.location.strip():
            extraction_stats['location_extracted'] += 1
    except Exception as e:
        logger.warning(f"   ⚠ Error extracting location from article {i}: {e}")
        article['stage3']['location'] = ""
        article['stage3']['location_reasoning'] = f"Error: {str(e)}"
        extraction_stats['errors'] += 1

    # Extract date
    try:
        date_pred = date_extractor(**example_input.inputs())
        article['stage3']['flood_date'] = date_pred.flood_date
        article['stage3']['date_reasoning'] = date_pred.reasoning

        if date_pred.flood_date.strip():
            extraction_stats['date_extracted'] += 1
    except Exception as e:
        logger.warning(f"   ⚠ Error extracting date from article {i}: {e}")
        article['stage3']['flood_date'] = ""
        article['stage3']['date_reasoning'] = f"Error: {str(e)}"
        extraction_stats['errors'] += 1

    # Check if both were extracted
    if (article['stage3'].get('location', '').strip() and
        article['stage3'].get('flood_date', '').strip()):
        extraction_stats['both_extracted'] += 1

    # Progress indicator
    if (i + 1) % progress_interval == 0:
        logger.info(f"  Processed {i+1:,}/{len(stage2_articles):,} articles... "
                   f"({extraction_stats['both_extracted']} complete)")

logger.info(f"\n✓ Extraction complete!")
logger.info(f"  Location extracted: {extraction_stats['location_extracted']:,} "
           f"({extraction_stats['location_extracted']/extraction_stats['total']:.1%})")
logger.info(f"  Date extracted: {extraction_stats['date_extracted']:,} "
           f"({extraction_stats['date_extracted']/extraction_stats['total']:.1%})")
logger.info(f"  Both extracted: {extraction_stats['both_extracted']:,} "
           f"({extraction_stats['both_extracted']/extraction_stats['total']:.1%})")
if extraction_stats['errors'] > 0:
    logger.warning(f"  Errors: {extraction_stats['errors']:,}")

# Save results
output_path = OUTPUT_DIR / 'stage3_extracted.json'
with open(output_path, 'w') as f:
    json.dump(stage2_articles, f, indent=2)
logger.info(f"\n✓ Results saved: {output_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

log_section(logger, "STAGE 3 COMPLETE")

logger.info(f"\nPipeline Summary:")
logger.info(f"  Stage 2 input (Ontario floods): {len(stage2_articles):,} articles")
logger.info(f"  After location extraction: {extraction_stats['location_extracted']:,} "
           f"({extraction_stats['location_extracted']/len(stage2_articles):.1%})")
logger.info(f"  After date extraction: {extraction_stats['date_extracted']:,} "
           f"({extraction_stats['date_extracted']/len(stage2_articles):.1%})")
logger.info(f"  Complete records (both): {extraction_stats['both_extracted']:,} "
           f"({extraction_stats['both_extracted']/len(stage2_articles):.1%})")

logger.info(f"\nOutput File:")
logger.info(f"  {output_path}")

log_section(logger, "Ready for Stage 4: Impact Extraction")
logger.info("")
