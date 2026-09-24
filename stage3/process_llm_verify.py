"""
Stage 3 LLM Verification: Verify NER locations and improve date extraction
Uses publication date context to resolve relative date mentions.

Run after process_ner.py to enhance extraction quality.

Features:
- Checkpoint/resume support for long-running processing
- Article ID normalization for traceability
- Shared logging configuration
- Parallel processing with thread safety
"""
import sys
from pathlib import Path
import json
import dspy
import re
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

# Import from shared utilities
from shared import (
    MODEL_CONFIG, PROJECT_ROOT, STAGE3_CONFIG,
    get_temperature, get_config_value,
    setup_logger, log_section, log_config,
    CheckpointManager,
    configure_dspy_lm,
    ensure_article_id, normalize_article_fields,
)

# Setup logging using shared config
logger = setup_logger(__name__, 'stage3_llm_verify', PROJECT_ROOT)

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
parser.add_argument('--resume', action='store_true',
                    help='Resume from checkpoint')
parser.add_argument('--clear-checkpoint', action='store_true',
                    help='Clear existing checkpoint and start fresh')
parser.add_argument('--checkpoint-interval', type=int, default=50,
                    help='Save checkpoint every N articles')
args = parser.parse_args()

# Set API key if provided
if args.api_key:
    import os
    os.environ['ANTHROPIC_API_KEY'] = args.api_key

log_section(logger, "STAGE 3: LLM VERIFICATION OF NER EXTRACTIONS")
print("\nVerifying locations and improving date extraction using LLM.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

NER_OUTPUT = PROJECT_ROOT / 'results' / (args.input or 'stage3_extracted_ner.json')
OUTPUT_DIR = PROJECT_ROOT / 'results'
CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

config_display = {
    'NER input': str(NER_OUTPUT),
    'LLM': MODEL_CONFIG['name'],
    'Resume mode': args.resume,
    'Checkpoint interval': args.checkpoint_interval,
}
if args.sample:
    config_display['Sample size'] = args.sample
log_config(logger, config_display)

# ============================================================================
# LOAD NER RESULTS
# ============================================================================

print("\n1. Loading NER extraction results...")

try:
    with open(NER_OUTPUT, 'r') as file:
        ner_articles = json.load(file)

    # Normalize article IDs for all articles
    for article in ner_articles:
        normalize_article_fields(article)

    logger.info(f"Loaded {len(ner_articles):,} articles with NER extractions")
except FileNotFoundError:
    logger.error(f"NER results not found at {NER_OUTPUT}")
    print(f"   Please run stage3/process_ner.py first")
    sys.exit(1)

# Apply sample limit if specified
if args.sample:
    ner_articles = ner_articles[:args.sample]
    logger.info(f"Processing sample of {len(ner_articles)} articles")

# ============================================================================
# INITIALIZE CHECKPOINT
# ============================================================================

print("\n2. Initializing checkpoint system...")

checkpoint = CheckpointManager(
    'stage3_llm_verify',
    CHECKPOINT_DIR,
    save_interval=args.checkpoint_interval
)

# Handle checkpoint modes
if args.clear_checkpoint:
    checkpoint.clear()
    logger.info("Cleared existing checkpoint")
elif args.resume:
    if checkpoint.load():
        logger.info(f"Resuming: {len(checkpoint.processed_ids)} already processed")

# ============================================================================
# CONFIGURE DSPY
# ============================================================================

print("\n3. Configuring LLM...")

# Configure language model using shared utility
temperature = get_temperature(STAGE3_CONFIG, mode='inference')
configure_dspy_lm(MODEL_CONFIG, temperature=temperature, mode='inference')

# Load verification signature
from stage3.signatures import FloodVerification

verifier = dspy.ChainOfThought(FloodVerification)
logger.info("FloodVerification signature loaded")

# ============================================================================
# VERIFY AND ENHANCE EXTRACTIONS
# ============================================================================

log_section(logger, "VERIFYING LOCATIONS AND EXTRACTING DATES")

# Filter to unprocessed articles if resuming
articles_to_process = []
article_indices = []
for i, article in enumerate(ner_articles):
    article_id = article.get('article_id')
    if args.resume and checkpoint.is_processed(article_id):
        continue
    articles_to_process.append(article)
    article_indices.append(i)

print(f"Processing {len(articles_to_process):,} articles...")
if len(articles_to_process) < len(ner_articles):
    print(f"  (Skipping {len(ner_articles) - len(articles_to_process)} already processed)")

# Get num_threads from config or args
num_threads = args.threads or get_config_value('num_threads', STAGE3_CONFIG) or 8
print(f"Using {num_threads} parallel threads\n")

# Pre-allocate results array with checkpoint data
verified_articles = [None] * len(ner_articles)

# Load already processed results from checkpoint
if args.resume:
    for result in checkpoint.get_results():
        # Find the index for this article
        article_id = result.get('article_id')
        for i, article in enumerate(ner_articles):
            if article.get('article_id') == article_id:
                verified_articles[i] = result
                break

stats = {
    'processed': len(checkpoint.processed_ids) if args.resume else 0,
    'location_verified': 0,
    'location_corrected': 0,
    'date_high_conf': 0,
    'date_medium_conf': 0,
    'date_low_conf': 0,
    'errors': 0
}
stats_lock = Lock()
checkpoint_lock = Lock()


def process_article(orig_idx, article):
    """Process a single article through LLM verification"""
    article_id = article.get('article_id')

    # Get NER results
    ner_stage3 = article.get('stage3', {})
    suggested_location = ner_stage3.get('location', 'not found')

    # Get publication date
    pub_date = article.get('date', article.get('publication_date', ''))

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

        # Thread-safe checkpoint update
        with checkpoint_lock:
            checkpoint.mark_processed(article_id, article)
            if checkpoint.should_save():
                checkpoint.save()

        return orig_idx, article, prediction.location_verified, prediction.date_confidence, None

    except Exception as e:
        logger.error(f"Error processing article {article_id}: {e}")
        # Keep NER results but mark as error
        article['stage3'] = {
            **ner_stage3,
            'method': 'NER (LLM error)',
            'llm_error': str(e)
        }

        with checkpoint_lock:
            checkpoint.mark_processed(article_id)

        return orig_idx, article, None, None, str(e)


# Process articles in parallel with progress bar
start_time = datetime.now()

if articles_to_process:
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all articles for processing
        futures = {executor.submit(process_article, article_indices[i], article): i
                   for i, article in enumerate(articles_to_process)}

        # Collect results as they complete with progress bar
        with tqdm(total=len(articles_to_process), desc="Verifying") as pbar:
            for future in as_completed(futures):
                orig_idx, article, loc_verified, date_conf, error = future.result()

                # Store in original order
                verified_articles[orig_idx] = article

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

# Final checkpoint save
checkpoint.save(force=True)

# Fill in any remaining None slots with original articles (shouldn't happen normally)
for i, article in enumerate(verified_articles):
    if article is None:
        verified_articles[i] = ner_articles[i]

elapsed_time = (datetime.now() - start_time).total_seconds()
articles_per_min = len(articles_to_process) / (elapsed_time / 60) if elapsed_time > 0 else 0

logger.info(f"Verification complete: {stats['processed']} processed, {stats['errors']} errors")
print(f"\n  Processing time: {elapsed_time/60:.1f} minutes ({articles_per_min:.0f} articles/min)")
print(f"\n  Location verification:")
print(f"    Verified (NER correct): {stats['location_verified']:,}")
print(f"    Corrected (NER wrong):  {stats['location_corrected']:,}")
print(f"\n  Date confidence:")
print(f"    High (explicit date):   {stats['date_high_conf']:,}")
print(f"    Medium (relative date): {stats['date_medium_conf']:,}")
print(f"    Low (estimated):        {stats['date_low_conf']:,}")
print(f"\n  Errors: {stats['errors']:,}")

# ============================================================================
# FILTER INVALID ARTICLES
# ============================================================================
# Remove articles that are not about specific Ontario flood events:
# 1. Location is "not found", "not applicable", or indicates no flood
# 2. Flood date could not be extracted (no year)
# 3. Location mentions non-Ontario places

print("\n" + "="*70)
print("FILTERING INVALID ARTICLES")
print("="*70)

def is_valid_flood_article(article):
    """Check if article describes a specific Ontario flood event."""
    stage3 = article.get('stage3', {})
    location = stage3.get('location', '').lower()
    flood_date = stage3.get('flood_date', '').lower()

    # Check location validity
    invalid_location_markers = [
        'not found', 'not applicable', 'unknown',
        'no flood', 'no ontario flood', 'not a flood'
    ]
    if any(marker in location for marker in invalid_location_markers):
        return False, 'invalid_location'

    # Check for non-Ontario locations in the location text
    non_ontario_keywords = [
        'manitoba', 'winnipeg', 'red river',
        'calgary', 'alberta', 'edmonton',
        'british columbia', 'vancouver',
        'saskatchewan', 'regina', 'saskatoon',
        'quebec', 'montreal', 'saguenay',
        'newfoundland', 'nova scotia', 'new brunswick', 'pei',
        'yukon', 'northwest territories', 'nunavut',
        'pakistan', 'india', 'china', 'bangladesh', 'costa rica',
        'united states', 'u.s.', 'usa', 'american'
    ]
    # Allow Ottawa-Gatineau (Ontario side)
    if 'gatineau' in location and 'ottawa' not in location:
        return False, 'non_ontario'
    for kw in non_ontario_keywords:
        if kw in location:
            return False, 'non_ontario'

    # Check flood date validity - must have at least a year
    invalid_date_markers = ['not found', 'not applicable', 'unknown', 'none']
    if any(marker in flood_date for marker in invalid_date_markers):
        return False, 'no_date'

    # Check if date contains a year (4-digit number)
    import re
    if not re.search(r'\d{4}', flood_date):
        return False, 'no_year'

    return True, 'valid'

# Apply filter
filter_stats = {
    'valid': 0,
    'invalid_location': 0,
    'non_ontario': 0,
    'no_date': 0,
    'no_year': 0
}

filtered_articles = []
rejected_articles = []

for article in verified_articles:
    is_valid, reason = is_valid_flood_article(article)
    filter_stats[reason] += 1
    if is_valid:
        filtered_articles.append(article)
    else:
        rejected_articles.append({
            'article_id': article.get('article_id'),
            'title': article.get('title', '')[:80],
            'reason': reason,
            'location': article.get('stage3', {}).get('location'),
            'flood_date': article.get('stage3', {}).get('flood_date')
        })

print(f"\nFilter results:")
print(f"  Valid flood articles:    {filter_stats['valid']:,}")
print(f"  Invalid location:        {filter_stats['invalid_location']:,}")
print(f"  Non-Ontario location:    {filter_stats['non_ontario']:,}")
print(f"  No flood date:           {filter_stats['no_date']:,}")
print(f"  No year in date:         {filter_stats['no_year']:,}")
print(f"  ─────────────────────────")
print(f"  Total removed:           {len(rejected_articles):,}")
print(f"  Kept for geocoding:      {len(filtered_articles):,}")

# Save rejected articles for review
rejected_path = OUTPUT_DIR / 'stage3_rejected.json'
with open(rejected_path, 'w') as f:
    json.dump(rejected_articles, f, indent=2)
print(f"\n✓ Rejected articles saved for review: {rejected_path}")

# Update verified_articles to only include valid ones
verified_articles = filtered_articles

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
    'total_articles_input': len(ner_articles),
    'total_articles_output': len(verified_articles),
    'processing_time_minutes': elapsed_time / 60,
    'articles_per_minute': articles_per_min,
    'verification_statistics': stats,
    'filter_statistics': filter_stats,
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
print(f"  Input articles:          {len(ner_articles):,}")
print(f"  Valid flood articles:    {len(verified_articles):,}")
print(f"  Filtered out:            {len(rejected_articles):,}")
print(f"\n  NER locations verified:  {stats['location_verified']:,} ({stats['location_verified']/max(stats['processed']-stats['errors'],1):.1%})")
print(f"  NER locations corrected: {stats['location_corrected']:,} ({stats['location_corrected']/max(stats['processed']-stats['errors'],1):.1%})")
print(f"  Date high/medium conf:   {stats['date_high_conf'] + stats['date_medium_conf']:,} ({(stats['date_high_conf'] + stats['date_medium_conf'])/max(stats['processed']-stats['errors'],1):.1%})")

print(f"\nOutput Files:")
print(f"  {output_path} ({len(verified_articles):,} valid articles)")
print(f"  {rejected_path} ({len(rejected_articles):,} rejected for review)")
print(f"  {summary_path}")

print(f"\n✅ Next steps:")
print(f"  1. Review rejected articles if needed: {rejected_path}")
print(f"  2. Geocode verified locations:")
print(f"     python stage3/geocode.py")

print("\n" + "="*70 + "\n")
