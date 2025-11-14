"""
Export detailed results to CSV for error analysis
"""
import sys
import json
import csv
from pathlib import Path
import dspy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import MODEL_CONFIG
from shared.utils import prepare_data
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
    test_data_raw = json.load(f)

print(f"\nLoaded {len(test_data_raw)} test examples")

# Convert to DSPy examples
test_data = prepare_data(test_data_raw)

# Load optimized model (or use baseline if not found)
# The optimize.py script saves models, let me check where
model_paths = [
    PROJECT_ROOT / 'models' / 'stage2_flood_verified.json',
    PROJECT_ROOT / 'stage2' / 'models' / 'flood_verifier.json',
    PROJECT_ROOT / 'models' / 'stage2' / 'flood_verifier.json',
]

optimized = None
for flood_model_path in model_paths:
    if flood_model_path.exists():
        print(f"Loading optimized model from {flood_model_path}")
        try:
            optimized = dspy.ChainOfThought(floodIdentification)
            optimized.load(str(flood_model_path))
            print("✓ Model loaded successfully")
            break
        except Exception as e:
            print(f"Error loading model: {e}")

if optimized is None:
    print("No saved model found, using baseline model")
    optimized = dspy.ChainOfThought(floodIdentification)

# Evaluate and collect results
print("\nEvaluating on test set...")
results = []

for i, (ex_raw, ex_dspy) in enumerate(zip(test_data_raw, test_data)):
    print(f"Processing {i+1}/{len(test_data)}...", end='\r')

    try:
        # Get prediction
        pred = optimized(
            article_text=ex_dspy.article_text,
            title=ex_dspy.title
        )

        true_label = ex_dspy.flood_mentioned
        pred_label = pred.flood_mentioned

        # Determine result type
        if true_label and pred_label:
            result_type = "True Positive"
        elif not true_label and not pred_label:
            result_type = "True Negative"
        elif not true_label and pred_label:
            result_type = "False Positive"
        else:  # true_label and not pred_label
            result_type = "False Negative"

        # Extract metadata
        result = {
            'id': ex_raw.get('id', ''),
            'title': ex_raw.get('title', ''),
            'publication_date': ex_raw.get('date', ''),
            'publisher': ex_raw.get('publisher', ''),
            'year': ex_raw.get('year', ''),
            'true_flood_mentioned': true_label,
            'predicted_flood_mentioned': pred_label,
            'result_type': result_type,
            'is_ontario': ex_raw.get('is_ontario', False),
            'flood_date': ex_raw.get('flood_date', ''),
            'model_reasoning': pred.reasoning if hasattr(pred, 'reasoning') else '',
            'article_text': ex_raw.get('full_text', '')[:500],  # First 500 chars
            'article_full': ex_raw.get('full_text', '')  # Full text
        }

        results.append(result)

    except Exception as e:
        print(f"\nError on example {i}: {e}")
        # Add error entry
        results.append({
            'id': ex_raw.get('id', ''),
            'title': ex_raw.get('title', ''),
            'publication_date': ex_raw.get('date', ''),
            'publisher': ex_raw.get('publisher', ''),
            'year': ex_raw.get('year', ''),
            'true_flood_mentioned': ex_dspy.flood_mentioned if hasattr(ex_dspy, 'flood_mentioned') else '',
            'predicted_flood_mentioned': 'ERROR',
            'result_type': 'ERROR',
            'is_ontario': ex_raw.get('is_ontario', False),
            'flood_date': ex_raw.get('flood_date', ''),
            'model_reasoning': f'Error: {str(e)}',
            'article_text': ex_raw.get('full_text', '')[:500],
            'article_full': ex_raw.get('full_text', '')
        })

print(f"\n✓ Processed {len(results)} examples")

# Calculate statistics
tp = sum(1 for r in results if r['result_type'] == 'True Positive')
tn = sum(1 for r in results if r['result_type'] == 'True Negative')
fp = sum(1 for r in results if r['result_type'] == 'False Positive')
fn = sum(1 for r in results if r['result_type'] == 'False Negative')
errors = sum(1 for r in results if r['result_type'] == 'ERROR')

print(f"\nResults Summary:")
print(f"  True Positives:  {tp}")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print(f"  Errors:          {errors}")
print(f"  Accuracy:        {(tp+tn)/len(results)*100:.1f}%")

# Export to CSV
output_file = PROJECT_ROOT / 'stage2' / 'results_analysis.csv'
print(f"\nExporting to {output_file}")

# CSV with preview text
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    fieldnames = [
        'id', 'title', 'publication_date', 'publisher', 'year',
        'true_flood_mentioned', 'predicted_flood_mentioned', 'result_type',
        'is_ontario', 'flood_date', 'model_reasoning', 'article_text'
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in results:
        # Write without full text
        row = {k: v for k, v in r.items() if k != 'article_full'}
        writer.writerow(row)

print(f"✓ Saved to {output_file}")

# Also export full text version
output_file_full = PROJECT_ROOT / 'stage2' / 'results_analysis_full.csv'
print(f"\nExporting full version to {output_file_full}")

with open(output_file_full, 'w', newline='', encoding='utf-8') as f:
    fieldnames = [
        'id', 'title', 'publication_date', 'publisher', 'year',
        'true_flood_mentioned', 'predicted_flood_mentioned', 'result_type',
        'is_ontario', 'flood_date', 'model_reasoning', 'article_full'
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for r in results:
        row = {
            'id': r['id'],
            'title': r['title'],
            'publication_date': r['publication_date'],
            'publisher': r['publisher'],
            'year': r['year'],
            'true_flood_mentioned': r['true_flood_mentioned'],
            'predicted_flood_mentioned': r['predicted_flood_mentioned'],
            'result_type': r['result_type'],
            'is_ontario': r['is_ontario'],
            'flood_date': r['flood_date'],
            'model_reasoning': r['model_reasoning'],
            'article_full': r['article_full']
        }
        writer.writerow(row)

print(f"✓ Saved full version to {output_file_full}")

# Create filtered CSVs for each error type
for error_type in ['False Positive', 'False Negative']:
    filtered = [r for r in results if r['result_type'] == error_type]
    if filtered:
        filename = error_type.lower().replace(' ', '_')
        output_file_filtered = PROJECT_ROOT / 'stage2' / f'{filename}_analysis.csv'

        with open(output_file_filtered, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id', 'title', 'publication_date', 'publisher', 'year',
                'true_flood_mentioned', 'predicted_flood_mentioned',
                'is_ontario', 'flood_date', 'model_reasoning', 'article_full'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in filtered:
                row = {
                    'id': r['id'],
                    'title': r['title'],
                    'publication_date': r['publication_date'],
                    'publisher': r['publisher'],
                    'year': r['year'],
                    'true_flood_mentioned': r['true_flood_mentioned'],
                    'predicted_flood_mentioned': r['predicted_flood_mentioned'],
                    'is_ontario': r['is_ontario'],
                    'flood_date': r['flood_date'],
                    'model_reasoning': r['model_reasoning'],
                    'article_full': r['article_full']
                }
                writer.writerow(row)

        print(f"✓ Saved {len(filtered)} {error_type} cases to {output_file_filtered}")

print("\n" + "="*70)
print("EXPORT COMPLETE")
print("="*70)
print(f"\nFiles created:")
print(f"  1. results_analysis.csv - All results with preview text")
print(f"  2. results_analysis_full.csv - All results with full article text")
print(f"  3. false_positive_analysis.csv - Only false positives ({fp} cases)")
print(f"  4. false_negative_analysis.csv - Only false negatives ({fn} cases)")
