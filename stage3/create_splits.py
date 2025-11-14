"""
Create train/test splits for Stage 3 from labeled flood data.

Stage 3 only uses flood-positive examples (is_ontario=True) since we're extracting
location and date information from verified Ontario floods.
"""
from pathlib import Path
import json
import random

# Set random seed for reproducibility
random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / 'data' / 'all_labeled_data.json'
OUTPUT_DIR = PROJECT_ROOT / 'stage3' / 'data'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CREATING STAGE 3 TRAIN/TEST SPLITS")
print("=" * 70)

# Load all labeled data
print(f"\nLoading data from: {INPUT_FILE}")
with open(INPUT_FILE, 'r') as f:
    all_data = json.load(f)

print(f"Total labeled examples: {len(all_data):,}")

# Filter to Ontario floods only (flood_mentioned=True AND is_ontario=True)
ontario_floods = [
    item for item in all_data
    if item.get('flood_mentioned', False) and item.get('is_ontario', False)
]

print(f"\nOntario flood examples: {len(ontario_floods):,} ({len(ontario_floods)/len(all_data):.1%})")

# Check how many have location and date annotations
has_location = sum(1 for item in ontario_floods if item.get('location', '').strip())
has_date = sum(1 for item in ontario_floods if item.get('flood_date', '').strip())
has_both = sum(
    1 for item in ontario_floods
    if item.get('location', '').strip() and item.get('flood_date', '').strip()
)

print(f"\nAnnotation coverage:")
print(f"  Has location: {has_location:,} ({has_location/len(ontario_floods):.1%})")
print(f"  Has flood_date: {has_date:,} ({has_date/len(ontario_floods):.1%})")
print(f"  Has both: {has_both:,} ({has_both/len(ontario_floods):.1%})")

# Shuffle the data
random.shuffle(ontario_floods)

# Split 70/30
split_idx = int(len(ontario_floods) * 0.7)
train_data = ontario_floods[:split_idx]
test_data = ontario_floods[split_idx:]

print(f"\nSplit:")
print(f"  Training: {len(train_data):,} examples ({len(train_data)/len(ontario_floods):.1%})")
print(f"  Test: {len(test_data):,} examples ({len(test_data)/len(ontario_floods):.1%})")

# Save splits
train_path = OUTPUT_DIR / 'stage3_train_70pct.json'
test_path = OUTPUT_DIR / 'stage3_test_30pct.json'

with open(train_path, 'w') as f:
    json.dump(train_data, f, indent=2)
print(f"\n✓ Training data saved: {train_path}")

with open(test_path, 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"✓ Test data saved: {test_path}")

print("\n" + "=" * 70)
print("SPLITS CREATED SUCCESSFULLY")
print("=" * 70)
print(f"\nReady for Stage 3 optimization with {len(ontario_floods)} Ontario flood examples")
print("=" * 70 + "\n")
