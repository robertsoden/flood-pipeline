"""
Create Stage 3 training/test splits from labeled data
Combines extraction reserve + stage2 train data with location/date annotations
"""
import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Load all labeled data sources
extraction_reserve = json.load(open(PROJECT_ROOT / 'stage1-bert' / 'data' / 'extraction_reserve_100.json'))
stage2_train = json.load(open(PROJECT_ROOT / 'stage2' / 'data' / 'stage2_train_70pct.json'))
stage2_test = json.load(open(PROJECT_ROOT / 'stage2' / 'data' / 'stage2_test_30pct.json'))

# Filter for Ontario floods with BOTH location AND date annotations
all_examples = []

for ex in extraction_reserve + stage2_train + stage2_test:
    if ex.get('flood_mentioned') and ex.get('is_ontario'):
        # Only include if we have both location AND date (non-empty strings)
        has_location = ex.get('location', '').strip() != ''
        has_date = ex.get('flood_date', '').strip() != ''
        if has_location and has_date:
            all_examples.append(ex)

print(f"Total examples with annotations: {len(all_examples)}")
print(f"  With location: {sum(1 for ex in all_examples if ex.get('location'))}")
print(f"  With flood_date: {sum(1 for ex in all_examples if ex.get('flood_date'))}")

# Split 70/30
random.seed(42)
random.shuffle(all_examples)

split_idx = int(len(all_examples) * 0.7)
train = all_examples[:split_idx]
test = all_examples[split_idx:]

print(f"\nSplit:")
print(f"  Train: {len(train)} examples")
print(f"  Test: {len(test)} examples")

# Save
output_dir = PROJECT_ROOT / 'stage3' / 'data'
output_dir.mkdir(exist_ok=True)

with open(output_dir / 'stage3_train_70pct.json', 'w') as f:
    json.dump(train, f, indent=2)

with open(output_dir / 'stage3_test_30pct.json', 'w') as f:
    json.dump(test, f, indent=2)

print(f"\n✓ Saved to stage3/data/")
