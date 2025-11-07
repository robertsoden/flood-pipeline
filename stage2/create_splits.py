"""
Create train/test splits for Stage 2 from all_labeled_data.json
This uses the complete labeled dataset with all metadata including titles.
"""
from pathlib import Path
import json
import random

# Set random seed for reproducibility
random.seed(42)

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_FILE = PROJECT_ROOT / 'data' / 'all_labeled_data.json'
OUTPUT_DIR = PROJECT_ROOT / 'stage2' / 'data'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CREATING STAGE 2 TRAIN/TEST SPLITS")
print("=" * 70)

# Load all labeled data
print(f"\nLoading data from: {INPUT_FILE}")
with open(INPUT_FILE, 'r') as f:
    all_data = json.load(f)

print(f"Total labeled examples: {len(all_data):,}")

# Shuffle the data
random.shuffle(all_data)

# Split 70/30
split_idx = int(len(all_data) * 0.7)
train_data = all_data[:split_idx]
test_data = all_data[split_idx:]

print(f"\nSplit:")
print(f"  Training: {len(train_data):,} examples ({len(train_data)/len(all_data):.1%})")
print(f"  Test: {len(test_data):,} examples ({len(test_data)/len(all_data):.1%})")

# Calculate label distribution
def get_label_stats(data):
    flood_count = sum(1 for item in data if item.get('flood_mentioned', False))
    ontario_count = sum(1 for item in data if item.get('is_ontario', False))
    return flood_count, ontario_count

train_floods, train_ontario = get_label_stats(train_data)
test_floods, test_ontario = get_label_stats(test_data)

print(f"\nTraining set distribution:")
print(f"  Flood mentions: {train_floods:,} ({train_floods/len(train_data):.1%})")
print(f"  Ontario floods: {train_ontario:,} ({train_ontario/len(train_data):.1%})")

print(f"\nTest set distribution:")
print(f"  Flood mentions: {test_floods:,} ({test_floods/len(test_data):.1%})")
print(f"  Ontario floods: {test_ontario:,} ({test_ontario/len(test_data):.1%})")

# Save splits
train_path = OUTPUT_DIR / 'stage2_train_70pct.json'
test_path = OUTPUT_DIR / 'stage2_test_30pct.json'

with open(train_path, 'w') as f:
    json.dump(train_data, f, indent=2)
print(f"\n✓ Training data saved: {train_path}")

with open(test_path, 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"✓ Test data saved: {test_path}")

print("\n" + "=" * 70)
print("SPLITS CREATED SUCCESSFULLY")
print("=" * 70)
print("\nNext step: Update shared/config.py to use these new files")
print("=" * 70 + "\n")
