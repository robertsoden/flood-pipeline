"""
Zero-Shot Classification Baseline for Flood Detection

Tests how well a pre-trained NLI model can detect flood articles
WITHOUT any fine-tuning on your data.

This provides a baseline to compare against your fine-tuned BERT model.

Usage:
    python zero-shot-baseline.py

Requirements:
    pip install transformers torch scikit-learn tqdm
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    recall_score
)
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Zero-shot model options (uncomment one):
    'model': 'facebook/bart-large-mnli',        # Best quality, slower
    # 'model': 'MoritzLaworski/DeBERTa-v3-small-zeroshot',  # Faster alternative
    # 'model': 'typeform/distilbert-base-uncased-mnli',     # Fastest, lower quality

    # Classification labels - adjust wording to improve results
    'positive_label': 'This article describes a flood event or flooding',
    'negative_label': 'This article is not about flooding',

    # Data paths (uses project structure, adjust if needed)
    'test_data_path': PROJECT_ROOT / 'stage2' / 'data' / 'stage2_test_30pct.json',

    # Processing
    'max_articles': None,  # Set to e.g. 100 for quick test, None for all
    'batch_size': 8,       # Reduce if memory issues

    # For comparison with fine-tuned model
    'target_recall': 0.95,
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_test_data(filepath):
    """Load test data and extract text + labels."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    articles = []
    for item in data:
        # Handle different data formats
        if 'annotations' in item:
            text = item.get('full_text', '')
            label = item['annotations'].get('flood_mentioned', False)
        else:
            text = item.get('article_text', item.get('full_text', ''))
            label = item.get('flood_mentioned', False)

        if text:  # Skip empty articles
            articles.append({
                'text': text,
                'label': 1 if label else 0
            })

    return articles


# =============================================================================
# ZERO-SHOT CLASSIFICATION
# =============================================================================

def run_zero_shot_classification(articles, config):
    """Run zero-shot classification on articles."""
    from transformers import pipeline

    print(f"\nLoading model: {config['model']}")
    print("(This may take a minute on first run to download the model...)\n")

    classifier = pipeline(
        "zero-shot-classification",
        model=config['model'],
        device=0 if __import__('torch').cuda.is_available() else -1
    )

    candidate_labels = [config['positive_label'], config['negative_label']]

    predictions = []
    probabilities = []

    # Process in batches for efficiency
    texts = [a['text'] for a in articles]
    batch_size = config['batch_size']

    print(f"Classifying {len(texts)} articles...")
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i + batch_size]

        # Truncate long texts (zero-shot models have token limits)
        batch = [t[:2000] for t in batch]  # Rough character limit

        results = classifier(batch, candidate_labels, multi_label=False)

        # Handle single result vs list
        if not isinstance(results, list):
            results = [results]

        for result in results:
            # Check if flood label won
            is_flood = result['labels'][0] == config['positive_label']
            flood_score = (
                result['scores'][0] if is_flood
                else result['scores'][1]
            )

            predictions.append(1 if is_flood else 0)
            probabilities.append(flood_score)

    return predictions, probabilities


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_predictions(true_labels, predictions, probabilities, config):
    """Calculate and print evaluation metrics."""

    print("\n" + "="*70)
    print("ZERO-SHOT CLASSIFICATION RESULTS")
    print("="*70)

    # Basic metrics at default threshold (0.5)
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='binary', zero_division=0
    )

    print(f"\nAt default threshold (classifier picks winner):")
    print(f"  Accuracy:  {accuracy:.1%}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1:        {f1:.3f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}, FN={fn}")
    print(f"    FP={fp}, TN={tn}")

    # Find threshold for target recall
    print(f"\n" + "-"*70)
    print(f"Finding threshold for {config['target_recall']:.0%}+ recall...")

    probs = np.array(probabilities)
    labels = np.array(true_labels)

    best_threshold = None
    best_precision = 0

    for threshold in np.arange(0.9, 0.1, -0.02):
        preds = (probs >= threshold).astype(int)
        rec = recall_score(labels, preds, zero_division=0)

        if rec >= config['target_recall']:
            prec = np.sum((labels == 1) & (preds == 1)) / np.sum(preds) if np.sum(preds) > 0 else 0
            if prec > best_precision:
                best_threshold = threshold
                best_precision = prec
                best_recall = rec

    if best_threshold:
        preds_at_threshold = (probs >= best_threshold).astype(int)
        tn2, fp2, fn2, tp2 = confusion_matrix(labels, preds_at_threshold).ravel()
        filter_rate = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

        print(f"\nAt threshold {best_threshold:.2f} (for {config['target_recall']:.0%}+ recall):")
        print(f"  Recall:      {best_recall:.1%}")
        print(f"  Precision:   {best_precision:.1%}")
        print(f"  Filter rate: {filter_rate:.1%}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP={tp2}, FN={fn2}")
        print(f"    FP={fp2}, TN={tn2}")
    else:
        print(f"\n  Could not achieve {config['target_recall']:.0%} recall")
        print(f"  Max recall achieved: {recall:.1%}")

    # Probability distribution
    print(f"\n" + "-"*70)
    print("Probability Distribution:")
    flood_probs = probs[labels == 1]
    non_flood_probs = probs[labels == 0]

    print(f"  Actual floods:     mean={flood_probs.mean():.2f}, min={flood_probs.min():.2f}, max={flood_probs.max():.2f}")
    print(f"  Actual non-floods: mean={non_flood_probs.mean():.2f}, min={non_flood_probs.min():.2f}, max={non_flood_probs.max():.2f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'threshold_for_target_recall': best_threshold,
        'precision_at_target_recall': best_precision if best_threshold else None,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = CONFIG

    print("="*70)
    print("ZERO-SHOT FLOOD DETECTION BASELINE")
    print("="*70)
    print(f"\nModel: {config['model']}")
    print(f"Positive label: \"{config['positive_label']}\"")
    print(f"Negative label: \"{config['negative_label']}\"")

    # Load data
    print(f"\nLoading test data from: {config['test_data_path']}")
    articles = load_test_data(config['test_data_path'])
    print(f"Loaded {len(articles)} articles")

    # Limit if requested
    if config['max_articles']:
        articles = articles[:config['max_articles']]
        print(f"Limited to {len(articles)} articles for testing")

    # Class distribution
    num_floods = sum(1 for a in articles if a['label'] == 1)
    print(f"\nClass distribution:")
    print(f"  Floods: {num_floods} ({num_floods/len(articles):.1%})")
    print(f"  Non-floods: {len(articles) - num_floods} ({(len(articles) - num_floods)/len(articles):.1%})")

    # Run classification
    predictions, probabilities = run_zero_shot_classification(articles, config)

    # Evaluate
    true_labels = [a['label'] for a in articles]
    results = evaluate_predictions(true_labels, predictions, probabilities, config)

    # Save results
    results_dir = PROJECT_ROOT / 'results'
    results_dir.mkdir(exist_ok=True)

    output_file = results_dir / 'zero_shot_baseline.json'
    with open(output_file, 'w') as f:
        json.dump({
            'model': config['model'],
            'positive_label': config['positive_label'],
            'negative_label': config['negative_label'],
            'num_articles': len(articles),
            'results': results,
        }, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Comparison note
    print("\n" + "="*70)
    print("COMPARISON WITH FINE-TUNED BERT")
    print("="*70)
    print("""
If your fine-tuned BERT achieves:
  - Recall: 97.7%, Precision: 76.8%

And zero-shot achieves (example):
  - Recall: 85%, Precision: 60%

Then fine-tuning is worth the effort for this task.

If zero-shot is close (within 5-10%), you might not need fine-tuning.
""")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
