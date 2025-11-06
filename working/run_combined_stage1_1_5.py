"""
Combined Stage 1 + 1.5: Flood Detection + Ontario Filtering
Runs both stages in one pass - outputs only Ontario floods
"""

# Suppress warnings
import os
import warnings
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
warnings.filterwarnings('ignore')

import json
import dspy
from tqdm import tqdm
import sys

import config
from signatures import floodIdentification, isOntario

# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================
INPUT_FILE = 'data/articles_restructured.json'  # Your input file
OUTPUT_ONTARIO = 'ontario_floods_only.json'  # Just Ontario floods
CHECKPOINT_INTERVAL = 500

# TEST MODE - Set to None for full 50K run
TEST_MODE = 180  # Process only first N articles (100, 200, etc.) or None for all

# ============================================================
# SCRIPT START
# ============================================================

print("\n" + "="*60)
if TEST_MODE:
    print(f"COMBINED FLOOD + ONTARIO DETECTION (TEST MODE)")
else:
    print("COMBINED FLOOD + ONTARIO DETECTION")
print("="*60)

# Check if models exist
if not os.path.exists('models/stage1_optimized.json'):
    print("\n❌ ERROR: models/stage1_optimized.json not found!")
    print("Train Stage 1 first: python stage1_flood_identification.py")
    sys.exit(1)

if not os.path.exists('models/stage1_5_optimized.json'):
    print("\n❌ ERROR: models/stage1_5_optimized.json not found!")
    print("Train Stage 1.5 first: python stage1_5_ontario_identification.py")
    sys.exit(1)

# Configure LM
print("\nConfiguring language model...")
lm = dspy.LM(
    config.MODEL_CONFIG['name'],
    api_base=config.MODEL_CONFIG['api_base'],
    api_key=config.MODEL_CONFIG['api_key']
)
dspy.configure(lm=lm)

# Load optimized models
print("Loading Stage 1 model (flood detection)...")
flood_extractor = dspy.ChainOfThought(floodIdentification)
flood_extractor.load('models/stage1_optimized.json')
print("✓ Stage 1 loaded")

print("Loading Stage 1.5 model (Ontario detection)...")
ontario_checker = dspy.ChainOfThought(isOntario)
ontario_checker.load('models/stage1_5_optimized.json')
print("✓ Stage 1.5 loaded")

# Load articles
if not os.path.exists(INPUT_FILE):
    print(f"\n❌ ERROR: {INPUT_FILE} not found!")
    sys.exit(1)

print(f"\nLoading articles from {INPUT_FILE}...")
with open(INPUT_FILE, 'r') as f:
    articles = json.load(f)

# Apply test mode if enabled
if TEST_MODE is not None:
    articles = articles[:TEST_MODE]
    print(f"⚡ TEST MODE: Processing only first {TEST_MODE} articles")

print(f"✓ Loaded {len(articles):,} articles")

# Estimate time
time_per_article = 15  # seconds (Stage 1)
time_per_flood = 15    # seconds (Stage 1.5, only on floods)
expected_floods = int(len(articles) * 0.27)  # Based on your 27% rate
expected_ontario = int(expected_floods * 0.15)  # ~15% of floods are Ontario

total_hours = (len(articles) * time_per_article + expected_floods * time_per_flood) / 3600
total_minutes = total_hours * 60

if TEST_MODE:
    print(f"\n⏱️  Estimated time: {total_minutes:.1f} minutes (TEST MODE)")
else:
    print(f"\n⏱️  Estimated time: {total_hours:.1f} hours (~{total_hours/24:.1f} days)")
    
print(f"   Stage 1: All {len(articles):,} articles → ~{expected_floods:,} floods")
print(f"   Stage 1.5: ~{expected_floods:,} floods → ~{expected_ontario:,} Ontario floods")

# Confirm before proceeding
if TEST_MODE:
    response = input(f"\nRun TEST on {len(articles):,} articles? (~{total_minutes:.0f} min) (yes/no): ")
else:
    response = input(f"\nProcess ALL {len(articles):,} articles? (~{total_hours:.1f} hours) (yes/no): ")
    
if response.lower() != 'yes':
    print("Cancelled.")
    sys.exit(0)

# Process articles
print("\n🔍 Processing articles...")
ontario_floods = []
total_processed = 0
flood_count = 0
ontario_count = 0

for i, article in enumerate(tqdm(articles, desc="Finding Ontario floods")):
    total_processed += 1
    
    # Extract text field
    text = article.get('full_text') or article.get('text') or article.get('article_text') or article.get('content')
    
    if not text:
        print(f"\n⚠️  Warning: Article {i} has no text field!")
        continue
    
    try:
        # STAGE 1: Check if flood mentioned
        stage1_pred = flood_extractor(article_text=text)
        
        if stage1_pred.flood_mentioned:
            flood_count += 1
            
            # STAGE 1.5: Check if Ontario (only for floods)
            try:
                stage1_5_pred = ontario_checker(article_text=text)
                
                # Only save if BOTH flood AND Ontario
                if stage1_5_pred.is_ontario:
                    ontario_count += 1
                    
                    # Keep complete original object
                    ontario_article = article.copy()
                    
                    # Add detection metadata
                    ontario_article['_flood_detection'] = {
                        'flood_mentioned': True,
                        'flood_reasoning': stage1_pred.reasoning,
                        'is_ontario': True,
                        'ontario_reasoning': stage1_5_pred.reasoning,
                        'detected_at': total_processed
                    }
                    
                    ontario_floods.append(ontario_article)
                    
            except Exception as e:
                print(f"\n⚠️  Stage 1.5 error on article {i}: {e}")
                # Don't add if Ontario check fails
        
    except Exception as e:
        print(f"\n⚠️  Stage 1 error on article {i}: {e}")
        continue
    
    # Save checkpoint
    if (i + 1) % CHECKPOINT_INTERVAL == 0:
        with open(f'ontario_checkpoint_{i+1}.json', 'w') as f:
            json.dump(ontario_floods, f, indent=2)
        
        print(f"\n✓ Checkpoint {i+1:,}: {flood_count:,} floods total, {ontario_count:,} Ontario floods ({ontario_count/flood_count if flood_count > 0 else 0:.1%} of floods)")

# Save final results
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

# Use different filename for test mode
output_file = f'ontario_floods_test_{len(articles)}.json' if TEST_MODE else OUTPUT_ONTARIO

with open(output_file, 'w') as f:
    json.dump(ontario_floods, f, indent=2)
print(f"\n✓ Ontario floods saved: {output_file} ({len(ontario_floods):,} articles)")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total articles processed: {total_processed:,}")
print(f"Floods detected (Stage 1): {flood_count:,} ({flood_count/total_processed:.1%})")
print(f"Ontario floods (Stage 1.5): {ontario_count:,} ({ontario_count/total_processed:.1%} of all, {ontario_count/flood_count if flood_count > 0 else 0:.1%} of floods)")

# Show examples
if len(ontario_floods) > 0:
    print(f"\n📰 First 3 Ontario flood detections:")
    print("-" * 60)
    for idx in range(min(3, len(ontario_floods))):
        article = ontario_floods[idx]
        text = article.get('full_text') or article.get('text', '')
        print(f"\nArticle {idx + 1}:")
        print(f"ID: {article.get('id', 'N/A')}")
        print(f"Date: {article.get('date', 'N/A')}")
        print(f"Title: {article.get('title', 'N/A')[:80]}...")
        print(f"Snippet: {text[:100]}...")

print("\n" + "="*60)
print("NEXT STEPS")
print("="*60)

if TEST_MODE:
    print(f"\n⚡ TEST RUN COMPLETE")
    print(f"✓ Found {ontario_count:,} Ontario floods in {len(articles):,} test articles")
    print(f"\nTo run on full 50K dataset:")
    print(f"  1. Edit script: Set TEST_MODE = None")
    print(f"  2. Run: python filter_floods_ontario_combined.py")
    print(f"  3. Wait ~11 days for completion")
else:
    print(f"\n✓ You have {ontario_count:,} Ontario flood articles in {output_file}")
    print(f"\nTo extract details (location, date, impacts):")
    print(f"  1. Train Stage 2: python stage2_detail_extraction.py")
    print(f"  2. Run detail extraction on the Ontario floods")
    
print("="*60)