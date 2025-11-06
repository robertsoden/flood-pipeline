"""
Inference Script: Apply Trained BERT to Complete Dataset
Classifies all unlabeled articles and generates filtered dataset
"""
import sys
from pathlib import Path
import json
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared config
from shared import unlabeled_filepath, STAGE1_CONFIG

print("\n" + "="*70)
print("BERT INFERENCE - CLASSIFY COMPLETE DATASET")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Get configuration from shared config
MODEL_PATH = STAGE1_CONFIG['bert_model_dir']
THRESHOLD_FILE = MODEL_PATH / 'threshold_info.json'
INPUT_FILE = unlabeled_filepath
OUTPUT_DIR = PROJECT_ROOT / 'results'
BATCH_SIZE = STAGE1_CONFIG.get('batch_size', 32)

print(f"\nConfiguration:")
print(f"  Model: {MODEL_PATH}")
print(f"  Input: {INPUT_FILE}")
print(f"  Output directory: {OUTPUT_DIR}")
print(f"  Batch size: {BATCH_SIZE}")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD MODEL AND THRESHOLD
# ============================================================================

print(f"\n1. Loading model and threshold...")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

print(f"   ✓ Model loaded on: {device}")

# Load optimal threshold
try:
    with open(THRESHOLD_FILE, 'r') as f:
        threshold_info = json.load(f)
        optimal_threshold = threshold_info['threshold']
        expected_recall = threshold_info.get('recall', 'N/A')
        expected_precision = threshold_info.get('precision', 'N/A')
    
    print(f"   ✓ Optimal threshold: {optimal_threshold:.3f}")
    print(f"   ✓ Expected recall: {expected_recall:.1%}" if expected_recall != 'N/A' else "")
    print(f"   ✓ Expected precision: {expected_precision:.1%}" if expected_precision != 'N/A' else "")
except FileNotFoundError:
    # Use threshold from config if available
    optimal_threshold = STAGE1_CONFIG.get('threshold', 0.30)
    print(f"   ⚠️  Threshold file not found, using config default: {optimal_threshold:.3f}")
    expected_recall = "Unknown"
    expected_precision = "Unknown"

# ============================================================================
# LOAD UNLABELED DATA
# ============================================================================

print(f"\n2. Loading unlabeled dataset...")

with open(INPUT_FILE, 'r') as f:
    unlabeled_data = json.load(f)

print(f"   ✓ Loaded {len(unlabeled_data):,} articles")

# ============================================================================
# CLASSIFY ALL ARTICLES
# ============================================================================

print(f"\n3. Classifying articles (this may take a while)...")

results = []
all_probabilities = []

# Process in batches
for i in tqdm(range(0, len(unlabeled_data), BATCH_SIZE), desc="Processing batches"):
    batch = unlabeled_data[i:i+BATCH_SIZE]
    batch_texts = [article.get('full_text', '') for article in batch]
    
    # Tokenize and predict
    with torch.no_grad():
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            truncation=True,
            max_length=STAGE1_CONFIG['max_length'],
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
    
    # Store results for this batch
    for idx, (article, prob) in enumerate(zip(batch, probs)):
        prediction = prob > optimal_threshold
        
        result = {
            'article_id': article.get('id', f'article_{i+idx}'),
            'title': article.get('title', ''),
            'date': article.get('publication_date', article.get('date', '')),
            'full_text': article.get('full_text', ''),
            'flood_probability': float(prob),
            'predicted_flood': bool(prediction),
            'confidence': 'HIGH' if abs(prob - 0.5) > 0.3 else 'MEDIUM' if abs(prob - 0.5) > 0.2 else 'LOW'
        }
        
        results.append(result)
        all_probabilities.append(prob)

print(f"\n   ✓ Classification complete!")

# ============================================================================
# ANALYZE RESULTS
# ============================================================================

print(f"\n4. Analyzing results...")

all_probabilities = np.array(all_probabilities)
predictions = all_probabilities > optimal_threshold

num_predicted_floods = np.sum(predictions)
num_predicted_non_floods = len(predictions) - num_predicted_floods

print(f"\n   Classification Summary:")
print(f"   {'='*60}")
print(f"   Total articles: {len(results):,}")
print(f"   Predicted floods: {num_predicted_floods:,} ({num_predicted_floods/len(results):.1%})")
print(f"   Predicted non-floods: {num_predicted_non_floods:,} ({num_predicted_non_floods/len(results):.1%})")
print(f"   Filter rate: {num_predicted_non_floods/len(results):.1%}")

print(f"\n   Probability Distribution:")
print(f"   {'='*60}")
print(f"   Mean probability: {all_probabilities.mean():.3f}")
print(f"   Median probability: {np.median(all_probabilities):.3f}")
print(f"   Std dev: {all_probabilities.std():.3f}")
print(f"   Min: {all_probabilities.min():.3f}")
print(f"   Max: {all_probabilities.max():.3f}")

# Confidence breakdown
high_conf = sum(1 for r in results if r['confidence'] == 'HIGH')
med_conf = sum(1 for r in results if r['confidence'] == 'MEDIUM')
low_conf = sum(1 for r in results if r['confidence'] == 'LOW')

print(f"\n   Confidence Levels:")
print(f"   {'='*60}")
print(f"   HIGH confidence: {high_conf:,} ({high_conf/len(results):.1%})")
print(f"   MEDIUM confidence: {med_conf:,} ({med_conf/len(results):.1%})")
print(f"   LOW confidence: {low_conf:,} ({low_conf/len(results):.1%})")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print(f"\n5. Saving results...")

# Save complete results (all articles with predictions)
output_complete = OUTPUT_DIR / 'all_articles_classified.json'
with open(output_complete, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   ✓ Complete results: {output_complete}")

# Save predicted floods only (for manual review)
floods_only = [r for r in results if r['predicted_flood']]
output_floods = OUTPUT_DIR / 'predicted_floods.json'
with open(output_floods, 'w') as f:
    json.dump(floods_only, f, indent=2)
print(f"   ✓ Predicted floods only: {output_floods}")

# Save filtered non-floods (articles we can skip)
non_floods_only = [r for r in results if not r['predicted_flood']]
output_non_floods = OUTPUT_DIR / 'filtered_non_floods.json'
with open(output_non_floods, 'w') as f:
    json.dump(non_floods_only, f, indent=2)
print(f"   ✓ Filtered non-floods: {output_non_floods}")

# Save summary CSV
df_summary = pd.DataFrame([
    {
        'article_id': r['article_id'],
        'title': r['title'][:100],  # Truncate long titles
        'date': r['date'],
        'probability': r['flood_probability'],
        'predicted_flood': r['predicted_flood'],
        'confidence': r['confidence']
    }
    for r in results
])
output_csv = OUTPUT_DIR / 'classification_summary.csv'
df_summary.to_csv(output_csv, index=False)
print(f"   ✓ Summary CSV: {output_csv}")

# Save statistics
stats = {
    'model_path': str(MODEL_PATH),
    'threshold': optimal_threshold,
    'expected_recall': expected_recall,
    'expected_precision': expected_precision,
    'total_articles': len(results),
    'predicted_floods': int(num_predicted_floods),
    'predicted_non_floods': int(num_predicted_non_floods),
    'flood_rate': float(num_predicted_floods / len(results)),
    'filter_rate': float(num_predicted_non_floods / len(results)),
    'mean_probability': float(all_probabilities.mean()),
    'median_probability': float(np.median(all_probabilities)),
    'high_confidence': int(high_conf),
    'medium_confidence': int(med_conf),
    'low_confidence': int(low_conf)
}

output_stats = OUTPUT_DIR / 'classification_stats.json'
with open(output_stats, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"   ✓ Statistics: {output_stats}")

# ============================================================================
# SUMMARY
# ============================================================================

print(f"\n{'='*70}")
print("CLASSIFICATION COMPLETE")
print(f"{'='*70}")

print(f"\n📊 Results:")
print(f"   Total articles processed: {len(results):,}")
print(f"   Predicted floods (to review): {num_predicted_floods:,}")
print(f"   Filtered non-floods (can skip): {num_predicted_non_floods:,}")
print(f"   Filter rate: {num_predicted_non_floods/len(results):.1%}")

print(f"\n💰 Estimated Savings:")
filtered_cost = num_predicted_non_floods * 0.01  # $0.01 per article
print(f"   Articles filtered: {num_predicted_non_floods:,}")
print(f"   Cost saved: ${filtered_cost:,.2f} (@ $0.01/article)")

print(f"\n📁 Output Files:")
print(f"   1. {output_complete}")
print(f"      → All {len(results):,} articles with predictions")
print(f"   ")
print(f"   2. {output_floods}")
print(f"      → {num_predicted_floods:,} predicted floods (REVIEW THESE)")
print(f"   ")
print(f"   3. {output_non_floods}")
print(f"      → {num_predicted_non_floods:,} filtered non-floods (CAN SKIP)")
print(f"   ")
print(f"   4. {output_csv}")
print(f"      → Summary spreadsheet")
print(f"   ")
print(f"   5. {output_stats}")
print(f"      → Classification statistics")

if expected_recall != "Unknown" and expected_precision != "Unknown":
    print(f"\n⚠️  Expected Performance:")
    print(f"   Based on test set:")
    print(f"   - Will catch ~{expected_recall:.1%} of actual floods")
    print(f"   - ~{expected_precision:.1%} of predicted floods are real floods")
    if isinstance(expected_precision, float):
        false_alarms = int(num_predicted_floods * (1 - expected_precision))
        print(f"   - May have ~{false_alarms:,} false alarms")

print(f"\n✅ Next Steps:")
print(f"   1. Review predicted floods: {output_floods}")
print(f"   2. Extract data (date, location, impacts) from floods")
print(f"   3. Spot-check some filtered non-floods for quality assurance")
print(f"   4. Compare results with your expectations")

print(f"\n{'='*70}\n")