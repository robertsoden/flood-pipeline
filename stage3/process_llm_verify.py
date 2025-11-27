"""
Stage 3 LLM Verification: Verify NER locations and improve date extraction
Uses publication date context to resolve relative date mentions.

Run after process_ner.py to enhance extraction quality.
"""
import sys
from pathlib import Path
import json
import dspy
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# Setup logging
LOGS_DIR = PROJECT_ROOT / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
log_file = LOGS_DIR / f'stage3_llm_verify_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

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

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='LLM verification of NER extractions')
parser.add_argument('--sample', type=int, default=None,
                    help='Process only first N articles (for testing)')
parser.add_argument('--threads', type=int, default=None,
                    help='Number of parallel threads (default from config)')
parser.add_argument('--input', type=str, default=None,
                    help='Input file (default: stage3_extracted_ner.json)')
parser.add_argument('--api-key', type=str, default=None,
                    help='Anthropic API key (or set ANTHROPIC_API_KEY env var)')
args = parser.parse_args()

# Set API key if provided
if args.api_key:
    import os
    os.environ['ANTHROPIC_API_KEY'] = args.api_key

print("\n" + "="*70)
print("STAGE 3: LLM VERIFICATION OF NER EXTRACTIONS")
print("="*70)
print("\nVerifying locations and improving date extraction using LLM.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

NER_OUTPUT = PROJECT_ROOT / 'results' / (args.input or 'stage3_extracted_ner.json')
OUTPUT_DIR = PROJECT_ROOT / 'results'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  NER input: {NER_OUTPUT}")
print(f"  LLM: {MODEL_CONFIG['name']}")
if args.sample:
    print(f"  Sample size: {args.sample} articles")

# ============================================================================
# LOAD NER RESULTS
# ============================================================================

print("\n1. Loading NER extraction results...")

try:
    with open(NER_OUTPUT, 'r') as file:
        ner_articles = json.load(file)
    print(f"   ✓ Loaded {len(ner_articles):,} articles with NER extractions")
except FileNotFoundError:
    print(f"   ❌ ERROR: NER results not found at {NER_OUTPUT}")
    print(f"   Please run stage3/process_ner.py first")
    sys.exit(1)

# Apply sample limit if specified
if args.sample:
    ner_articles = ner_articles[:args.sample]
    print(f"   ℹ️  Processing sample of {len(ner_articles)} articles")

# ============================================================================
# CONFIGURE DSPY
# ============================================================================

print("\n2. Configuring LLM...")

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

# Load verification signature
from stage3.signatures import FloodVerification

verifier = dspy.ChainOfThought(FloodVerification)
print(f"   ✓ FloodVerification signature loaded")

# ============================================================================
# VERIFY AND ENHANCE EXTRACTIONS
# ============================================================================

print("\n" + "="*70)
print("VERIFYING LOCATIONS AND EXTRACTING DATES")
print("="*70)
print(f"Processing {len(ner_articles):,} articles...")

# Get num_threads from config or args
num_threads = args.threads or get_config_value('num_threads', STAGE3_CONFIG) or 8
print(f"Using {num_threads} parallel threads\n")

verified_articles = [None] * len(ner_articles)  # Pre-allocate to maintain order
stats = {
    'processed': 0,
    'location_verified': 0,
    'location_corrected': 0,
    'date_high_conf': 0,
    'date_medium_conf': 0,
    'date_low_conf': 0,
    'errors': 0
}
stats_lock = Lock()


def process_article(i, article):
    """Process a single article through LLM verification"""
    # Get NER results
    ner_stage3 = article.get('stage3', {})
    suggested_location = ner_stage3.get('location', 'not found')

    # Get publication date
    pub_date = article.get('date', '')

    # Create DSPy example
    try:
        prediction = verifier(
            article_text=article.get('full_text', ''),
            title=article.get('title', ''),
            publication_date=pub_date,
            suggested_location=suggested_location
        )

        # Update article with verified results
        article['stage3'] = {
            'location': prediction.location,
            'location_verified': prediction.location_verified,
            'location_ner_suggested': suggested_location,
            'flood_date': prediction.flood_date,
            'date_confidence': prediction.date_confidence,
            'reasoning': prediction.reasoning,
            'method': 'NER+LLM',
            'publication_date': pub_date
        }

        return i, article, prediction.location_verified, prediction.date_confidence, None

    except Exception as e:
        logger.error(f"Error processing article {i}: {e}")
        # Keep NER results but mark as error
        article['stage3'] = {
            **ner_stage3,
            'method': 'NER (LLM error)',
            'llm_error': str(e)
        }
        return i, article, None, None, str(e)


# Process articles in parallel with progress bar
start_time = datetime.now()

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Submit all articles for processing
    futures = {executor.submit(process_article, i, article): i
               for i, article in enumerate(ner_articles)}

    # Collect results as they complete with progress bar
    with tqdm(total=len(ner_articles), desc="Verifying") as pbar:
        for future in as_completed(futures):
            i, article, loc_verified, date_conf, error = future.result()

            # Store in original order
            verified_articles[i] = article

            # Update statistics (thread-safe)
            with stats_lock:
                stats['processed'] += 1

                if error:
                    stats['errors'] += 1
                else:
                    if loc_verified:
                        stats['location_verified'] += 1
                    else:
                        stats['location_corrected'] += 1

                    if date_conf == 'high':
                        stats['date_high_conf'] += 1
                    elif date_conf == 'medium':
                        stats['date_medium_conf'] += 1
                    else:
                        stats['date_low_conf'] += 1

                pbar.update(1)

elapsed_time = (datetime.now() - start_time).total_seconds()
articles_per_min = len(ner_articles) / (elapsed_time / 60) if elapsed_time > 0 else 0

print(f"\n✓ Verification complete!")
print(f"  Processing time: {elapsed_time/60:.1f} minutes ({articles_per_min:.0f} articles/min)")
print(f"\n  Location verification:")
print(f"    Verified (NER correct): {stats['location_verified']:,}")
print(f"    Corrected (NER wrong):  {stats['location_corrected']:,}")
print(f"\n  Date confidence:")
print(f"    High (explicit date):   {stats['date_high_conf']:,}")
print(f"    Medium (relative date): {stats['date_medium_conf']:,}")
print(f"    Low (estimated):        {stats['date_low_conf']:,}")
print(f"\n  Errors: {stats['errors']:,}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Determine output filename
if args.sample:
    output_path = OUTPUT_DIR / f'stage3_verified_sample_{args.sample}.json'
else:
    output_path = OUTPUT_DIR / 'stage3_verified.json'

with open(output_path, 'w') as f:
    json.dump(verified_articles, f, indent=2)
print(f"\n✓ Results saved: {output_path}")

# Save summary statistics
summary = {
    'method': 'NER + LLM verification',
    'model': MODEL_CONFIG['name'],
    'total_articles': len(ner_articles),
    'processing_time_minutes': elapsed_time / 60,
    'articles_per_minute': articles_per_min,
    'statistics': stats,
    'verification_rates': {
        'location_verified_rate': stats['location_verified'] / max(stats['processed'] - stats['errors'], 1),
        'location_corrected_rate': stats['location_corrected'] / max(stats['processed'] - stats['errors'], 1),
        'date_high_conf_rate': stats['date_high_conf'] / max(stats['processed'] - stats['errors'], 1),
        'date_medium_conf_rate': stats['date_medium_conf'] / max(stats['processed'] - stats['errors'], 1),
        'date_low_conf_rate': stats['date_low_conf'] / max(stats['processed'] - stats['errors'], 1),
    }
}

if args.sample:
    summary_path = OUTPUT_DIR / f'stage3_verified_sample_{args.sample}_summary.json'
else:
    summary_path = OUTPUT_DIR / 'stage3_verified_summary.json'

with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {summary_path}")

# ============================================================================
# SAMPLE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAMPLE RESULTS")
print("="*70)

# Show first 5 results with interesting cases
count = 0
for article in verified_articles:
    if count >= 5:
        break
    stage3 = article.get('stage3', {})
    if stage3.get('method') == 'NER+LLM':
        count += 1
        print(f"\n{count}. {article.get('title', '')[:60]}...")
        print(f"   Publication date: {stage3.get('publication_date', 'N/A')}")
        print(f"   NER suggested:    {stage3.get('location_ner_suggested', 'N/A')}")
        print(f"   LLM location:     {stage3['location']} (verified: {stage3['location_verified']})")
        print(f"   Flood date:       {stage3['flood_date']} ({stage3['date_confidence']} confidence)")
        print(f"   Reasoning:        {stage3['reasoning'][:100]}...")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("STAGE 3 LLM VERIFICATION COMPLETE")
print("="*70)

print(f"\nResults Summary:")
print(f"  Total articles: {len(ner_articles):,}")
print(f"  NER locations verified: {stats['location_verified']:,} ({stats['location_verified']/max(stats['processed']-stats['errors'],1):.1%})")
print(f"  NER locations corrected: {stats['location_corrected']:,} ({stats['location_corrected']/max(stats['processed']-stats['errors'],1):.1%})")
print(f"  Date extraction: {stats['date_high_conf'] + stats['date_medium_conf']:,} high/medium confidence ({(stats['date_high_conf'] + stats['date_medium_conf'])/max(stats['processed']-stats['errors'],1):.1%})")

print(f"\nOutput Files:")
print(f"  {output_path}")
print(f"  {summary_path}")

print(f"\n✅ Next steps:")
print(f"  1. Review sample results above")
print(f"  2. Geocode verified locations:")
print(f"     python stage3/geocode.py")

print("\n" + "="*70 + "\n")
