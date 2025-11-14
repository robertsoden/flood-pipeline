"""
Analyze errors from the optimized Stage 2 model
"""
import sys
import json
from pathlib import Path
import dspy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import MODEL_CONFIG, STAGE2_CONFIG
from stage2.signatures import floodIdentification

# Configure LLM
lm = dspy.LM(
    model=MODEL_CONFIG['name'],
    api_base=MODEL_CONFIG['api_base'],
    api_key=MODEL_CONFIG['api_key'],
    temperature=0.1  # Use inference temperature
)
dspy.configure(lm=lm)

# Load test data
test_file = PROJECT_ROOT / 'stage2' / 'data' / 'stage2_test_30pct.json'
with open(test_file) as f:
    test_data = json.load(f)

print(f"\nLoaded {len(test_data)} test examples")

# Load optimized model
model_path = PROJECT_ROOT / 'stage2' / 'models' / 'flood_verifier.json'
print(f"Loading optimized model from {model_path}")

try:
    optimized = dspy.ChainOfThought(floodIdentification)
    optimized.load(str(model_path))
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Using baseline model instead")
    optimized = dspy.ChainOfThought(floodIdentification)

# Evaluate and collect errors
print("\nEvaluating on test set...")
fps = []  # False positives
fns = []  # False negatives
correct = 0

for ex in test_data:
    try:
        pred = optimized(
            article_text=ex['article_text'],
            title=ex['title']
        )

        true_label = ex['flood_mentioned']
        pred_label = pred.flood_mentioned

        if true_label == pred_label:
            correct += 1
        elif not true_label and pred_label:
            # False positive (said flood, wasn't)
            fps.append((ex, pred))
        elif true_label and not pred_label:
            # False negative (said not flood, was)
            fns.append((ex, pred))
    except Exception as e:
        print(f"Error on example: {e}")

total = len(test_data)
accuracy = correct / total * 100

print("\n" + "="*70)
print("ERROR ANALYSIS")
print("="*70)
print(f"\n📊 Performance:")
print(f"   Total test examples: {total}")
print(f"   Correct: {correct} ({accuracy:.1f}%)")
print(f"   False Positives: {len(fps)} ({len(fps)/total*100:.1f}%)")
print(f"   False Negatives: {len(fns)} ({len(fns)/total*100:.1f}%)")

# Analyze false positive patterns
print(f"\n🔴 FALSE POSITIVES ({len(fps)} total) - Model said FLOOD, actually NOT:")

fp_patterns = {'metaphorical': 0, 'planning': 0, 'historical': 0, 'warning': 0, 'other': 0}
for ex, pred in fps:
    text = ex['article_text'].lower()
    if 'flood of' in text or 'flooded with' in text:
        fp_patterns['metaphorical'] += 1
    elif 'insurance' in text or 'planning' in text or 'prepare for' in text:
        fp_patterns['planning'] += 1
    elif 'anniversary' in text or 'years ago' in text or 'commemorate' in text:
        fp_patterns['historical'] += 1
    elif 'flood warning' in text or 'flood watch' in text or 'flood advisory' in text:
        fp_patterns['warning'] += 1
    else:
        fp_patterns['other'] += 1

print(f"\n   Pattern breakdown:")
print(f"     Metaphorical floods: {fp_patterns['metaphorical']}/{len(fps)}")
print(f"     Planning/insurance: {fp_patterns['planning']}/{len(fps)}")
print(f"     Historical mentions: {fp_patterns['historical']}/{len(fps)}")
print(f"     Warnings only: {fp_patterns['warning']}/{len(fps)}")
print(f"     Other: {fp_patterns['other']}/{len(fps)}")

print(f"\n   Sample false positives:")
for i, (ex, pred) in enumerate(fps[:10], 1):
    print(f"\n   {i}. Title: {ex['title'][:80]}")
    print(f"      Text: {ex['article_text'][:200]}...")
    print(f"      Model reasoning: {pred.reasoning[:150]}...")

# Analyze false negatives
print(f"\n🔵 FALSE NEGATIVES ({len(fns)} total) - Model said NOT FLOOD, actually WAS:")

print(f"\n   Sample false negatives:")
for i, (ex, pred) in enumerate(fns[:10], 1):
    print(f"\n   {i}. Title: {ex['title'][:80]}")
    print(f"      Text: {ex['article_text'][:200]}...")
    print(f"      Model reasoning: {pred.reasoning[:150]}...")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)

# Calculate how many more correct we need for 88%
target = 0.88
current = correct / total
needed = int(target * total) - correct
print(f"\nTo reach {target*100}% accuracy:")
print(f"   Current: {correct}/{total} ({current*100:.1f}%)")
print(f"   Target: {int(target*total)}/{total} ({target*100:.1f}%)")
print(f"   Need {needed} more correct predictions")
print(f"   Must fix {min(len(fps), len(fns), needed)} errors (reducing both FP and FN)")
