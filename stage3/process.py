"""
Stage 3 Processing: Extract flood locations and dates from Ontario floods
Run this after Stage 2 to extract location and date information.
"""
import sys
from pathlib import Path
import json
import dspy
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f'stage3_process_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

# Import from shared config
from shared.config import MODEL_CONFIG, PROJECT_ROOT, STAGE3_CONFIG, get_temperature, get_config_value

print("\n" + "="*70)
print("STAGE 3: LOCATION & DATE EXTRACTION")
print("="*70)
print("\nExtracting flood locations and dates from Ontario floods.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

STAGE2_OUTPUT = PROJECT_ROOT / 'results' / 'stage2_ontario_floods.json'
OUTPUT_DIR = PROJECT_ROOT / 'results'
MODELS_DIR = PROJECT_ROOT / 'models'

EXTRACTION_MODEL_PATH = MODELS_DIR / 'stage3_location_extraction.json'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  Stage 2 input: {STAGE2_OUTPUT}")
print(f"  Extraction model: {EXTRACTION_MODEL_PATH}")
print(f"  LLM: {MODEL_CONFIG['name']}")

# ============================================================================
# CHECK FOR OPTIMIZED MODEL
# ============================================================================

print("\n1. Checking for optimized model...")

if not EXTRACTION_MODEL_PATH.exists():
    print(f"   ⚠️  WARNING: Extraction model not found!")
    print(f"   Will use baseline model (not optimized)")
    print(f"   For better results, run: python stage3/optimize.py")
    use_optimized = False
else:
    print(f"   ✓ Found optimized extraction model")
    use_optimized = True

# ============================================================================
# LOAD STAGE 2 RESULTS
# ============================================================================

print("\n2. Loading Stage 2 Ontario floods...")

try:
    with open(STAGE2_OUTPUT, 'r') as file:
        ontario_floods = json.load(file)
    print(f"   ✓ Loaded {len(ontario_floods):,} Ontario flood articles")
except FileNotFoundError:
    print(f"   ❌ ERROR: Stage 2 results not found at {STAGE2_OUTPUT}")
    print(f"   Please run stage2/process.py first")
    sys.exit(1)

# ============================================================================
# CONFIGURE DSPY AND LOAD MODEL
# ============================================================================

print("\n3. Loading extraction model...")

# Configure language model with inference temperature
temperature = get_temperature(STAGE3_CONFIG, mode='inference')

# Build LM kwargs
lm_kwargs = {'temperature': temperature}
if MODEL_CONFIG.get('api_base'):
    lm_kwargs['api_base'] = MODEL_CONFIG['api_base']
if MODEL_CONFIG.get('api_key'):
    lm_kwargs['api_key'] = MODEL_CONFIG['api_key']

lm = dspy.LM(MODEL_CONFIG['name'], **lm_kwargs)
dspy.configure(lm=lm)
print(f"   ✓ LM configured: {MODEL_CONFIG['name']} (temperature={temperature})")

# Load extraction model
from stage3.signatures import FloodLocationExtraction

extractor = dspy.ChainOfThought(FloodLocationExtraction)
if use_optimized:
    extractor.load(str(EXTRACTION_MODEL_PATH))
    print(f"   ✓ Loaded optimized extraction model")
else:
    print(f"   ℹ️  Using baseline model (not optimized)")

# ============================================================================
# EXTRACT LOCATIONS AND DATES
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING LOCATIONS AND DATES")
print("="*70)
print(f"Processing {len(ontario_floods):,} Ontario flood articles...")

# Get num_threads from config (default to 8 as specified in STAGE3_CONFIG)
num_threads = get_config_value('num_threads', STAGE3_CONFIG) or 8
print(f"Using {num_threads} parallel threads for extraction\n")

progress_interval = get_config_value('progress_interval', STAGE3_CONFIG)
extracted_articles = [None] * len(ontario_floods)  # Pre-allocate to maintain order
stats = {'extracted': 0, 'no_location': 0, 'no_date': 0, 'processed': 0}
stats_lock = Lock()  # Thread-safe statistics

def process_article(i, article):
    """Process a single article and return results"""
    # Create DSPy example
    example_input = dspy.Example(
        title=article.get('title', ''),
        article_text=article.get('full_text', '')
    ).with_inputs('title', 'article_text')

    # Extract
    try:
        prediction = extractor(**example_input.inputs())

        # Add Stage 3 results to article
        article['stage3'] = {
            'location': prediction.location,
            'flood_date': prediction.flood_date,
            'reasoning': prediction.reasoning,
        }

        # Calculate stats
        has_location = prediction.location and prediction.location.lower() not in ['unknown', 'not specified', 'none']
        has_date = prediction.flood_date and prediction.flood_date.lower() not in ['unknown', 'not specified', 'none']

        return i, article, has_location, has_date, None

    except Exception as e:
        logger.error(f"Error processing article {i}: {e}")
        # Keep article but mark as unprocessed
        article['stage3'] = {
            'location': 'ERROR',
            'flood_date': 'ERROR',
            'reasoning': f"Error: {str(e)}",
        }
        return i, article, False, False, str(e)

# Process articles in parallel
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all articles for processing
    futures = {executor.submit(process_article, i, article): i
               for i, article in enumerate(ontario_floods)}

    # Collect results as they complete
    for future in as_completed(futures):
        i, article, has_location, has_date, error = future.result()

        # Store in original order
        extracted_articles[i] = article

        # Update statistics (thread-safe)
        with stats_lock:
            stats['processed'] += 1
            if has_location:
                stats['extracted'] += 1
            else:
                stats['no_location'] += 1
            if not has_date:
                stats['no_date'] += 1

            # Progress indicator
            if stats['processed'] % progress_interval == 0:
                print(f"  Processed {stats['processed']:,}/{len(ontario_floods):,} articles... "
                      f"({stats['extracted']} locations extracted)")

print(f"\n✓ Extraction complete!")
print(f"  Locations extracted: {stats['extracted']:,} ({stats['extracted']/len(ontario_floods):.1%})")
print(f"  No location found: {stats['no_location']:,}")
print(f"  No date found: {stats['no_date']:,}")

# Save extracted data
output_path = OUTPUT_DIR / 'stage3_extracted_locations.json'
with open(output_path, 'w') as f:
    json.dump(extracted_articles, f, indent=2)
print(f"✓ Extracted data saved: {output_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("STAGE 3 COMPLETE")
print("="*70)

print(f"\nPipeline Summary:")
print(f"  Stage 2 input (Ontario floods): {len(ontario_floods):,} articles")
print(f"  Locations extracted: {stats['extracted']:,} articles")

print(f"\nOutput File:")
print(f"  {output_path}")

print(f"\n✅ Next step:")
print(f"  Geocode locations with Mapbox:")
print(f"  python stage3/geocode.py")

print("\n" + "="*70 + "\n")
