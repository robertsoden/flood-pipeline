"""
Stage 3 NER Processing: Extract flood locations and dates using NER
Fast, local processing - no API calls needed
"""
import sys
from pathlib import Path
import json
import time
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.ner_extractor import extract_location_date

print("\n" + "="*70)
print("STAGE 3: NER-BASED LOCATION & DATE EXTRACTION")
print("="*70)
print("\nExtracting flood locations and dates using spaCy NER + regex.\n")

# ============================================================================
# CONFIGURATION
# ============================================================================

STAGE2_OUTPUT = PROJECT_ROOT / 'results' / 'stage2_ontario_floods.json'
OUTPUT_DIR = PROJECT_ROOT / 'results'

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Configuration:")
print(f"  Stage 2 input: {STAGE2_OUTPUT}")
print(f"  Method: spaCy NER + regex patterns")
print(f"  Cost: $0 (runs locally)")

# ============================================================================
# LOAD STAGE 2 RESULTS
# ============================================================================

print("\n1. Loading Stage 2 Ontario floods...")

try:
    with open(STAGE2_OUTPUT, 'r') as file:
        ontario_floods = json.load(file)
    print(f"   ✓ Loaded {len(ontario_floods):,} Ontario flood articles")
except FileNotFoundError:
    print(f"   ❌ ERROR: Stage 2 results not found at {STAGE2_OUTPUT}")
    print(f"   Please run stage2/process.py first")
    sys.exit(1)

# ============================================================================
# EXTRACT LOCATIONS AND DATES USING NER
# ============================================================================

print("\n" + "="*70)
print("EXTRACTING LOCATIONS AND DATES")
print("="*70)
print(f"Processing {len(ontario_floods):,} Ontario flood articles...")
print()

start_time = time.time()
extracted_articles = []
stats = {
    'extracted_location': 0,
    'extracted_date': 0,
    'extracted_both': 0,
    'no_location': 0,
    'no_date': 0
}

for article in tqdm(ontario_floods, desc="Processing"):
    title = article.get('title', '')
    text = article.get('full_text', '')

    # Extract using NER
    location, flood_date = extract_location_date(title, text)

    # Add Stage 3 results to article
    article['stage3'] = {
        'location': location if location else 'not found',
        'flood_date': flood_date if flood_date else 'not found',
        'method': 'NER',
    }

    # Track stats
    if location:
        stats['extracted_location'] += 1
    else:
        stats['no_location'] += 1

    if flood_date:
        stats['extracted_date'] += 1
    else:
        stats['no_date'] += 1

    if location and flood_date:
        stats['extracted_both'] += 1

    extracted_articles.append(article)

elapsed_time = time.time() - start_time
articles_per_min = len(ontario_floods) / (elapsed_time / 60)

print(f"\n✓ Extraction complete!")
print(f"  Processing time: {elapsed_time/60:.1f} minutes ({articles_per_min:.0f} articles/min)")
print(f"  Locations extracted: {stats['extracted_location']:,} ({stats['extracted_location']/len(ontario_floods):.1%})")
print(f"  Dates extracted: {stats['extracted_date']:,} ({stats['extracted_date']/len(ontario_floods):.1%})")
print(f"  Both extracted: {stats['extracted_both']:,} ({stats['extracted_both']/len(ontario_floods):.1%})")
print(f"  No location found: {stats['no_location']:,}")
print(f"  No date found: {stats['no_date']:,}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save all extracted data
output_path = OUTPUT_DIR / 'stage3_extracted_ner.json'
with open(output_path, 'w') as f:
    json.dump(extracted_articles, f, indent=2)
print(f"\n✓ Results saved: {output_path}")

# Save summary statistics
summary = {
    'method': 'NER (spaCy + regex)',
    'total_articles': len(ontario_floods),
    'processing_time_minutes': elapsed_time / 60,
    'articles_per_minute': articles_per_min,
    'cost': 0.0,
    'statistics': stats,
    'extraction_rates': {
        'location_rate': stats['extracted_location'] / len(ontario_floods),
        'date_rate': stats['extracted_date'] / len(ontario_floods),
        'both_rate': stats['extracted_both'] / len(ontario_floods),
    }
}

summary_path = OUTPUT_DIR / 'stage3_ner_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved: {summary_path}")

# ============================================================================
# SAMPLE RESULTS
# ============================================================================

print("\n" + "="*70)
print("SAMPLE RESULTS")
print("="*70)

# Show first 10 extractions
for i, article in enumerate(extracted_articles[:10], 1):
    stage3 = article['stage3']
    print(f"\n{i}. {article.get('title', '')[:70]}...")
    print(f"   Location: {stage3['location']}")
    print(f"   Date: {stage3['flood_date']}")

# ============================================================================
# NEXT STEPS
# ============================================================================

print("\n" + "="*70)
print("STAGE 3 NER COMPLETE")
print("="*70)

print(f"\nResults Summary:")
print(f"  Total articles: {len(ontario_floods):,}")
print(f"  Locations extracted: {stats['extracted_location']:,} ({stats['extracted_location']/len(ontario_floods):.1%})")
print(f"  Dates extracted: {stats['extracted_date']:,} ({stats['extracted_date']/len(ontario_floods):.1%})")
print(f"  Processing time: {elapsed_time/60:.1f} minutes")
print(f"  Cost: $0")

print(f"\nOutput Files:")
print(f"  {output_path}")
print(f"  {summary_path}")

print(f"\n✅ Next steps:")
print(f"  1. Review sample results above")
print(f"  2. Optionally run LLM validation on uncertain cases")
print(f"  3. Geocode locations with Mapbox:")
print(f"     python stage3/geocode.py")
print(f"  4. Deduplicate flood events")

print("\n" + "="*70 + "\n")
