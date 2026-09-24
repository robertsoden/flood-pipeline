#!/usr/bin/env python3
"""
Evaluate BERT Recall on Historical Newspaper Text

BERT's role in the pipeline is to FILTER OUT obviously non-flood articles.
We need HIGH RECALL - it's OK to let false positives through (Stage 2 LLM handles those).
The critical question: Does BERT miss actual floods in historical text?

This script tests recall by:
1. Finding articles with flood keywords (proxy for actual floods)
2. Running BERT inference
3. Checking if BERT lets those articles through (recall on keyword matches)

If recall is high (>90%), BERT is working as intended for historical text.
If recall is low, we're filtering out potential floods and need to retrain.

Usage:
    python scripts/evaluate_historical_bert.py [--sample-size 500]
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent

# Model path (avoiding shared module dependency)
MODEL_PATH = PROJECT_ROOT / 'models' / 'balanced_high_recall_iter0'

# Historical flood keywords - if an article contains these, it's likely flood-related
# BERT should NOT filter these out
HISTORICAL_FLOOD_KEYWORDS = [
    # 19th century terms (critical for historical accuracy)
    r'\bfreshet\b',
    r'\binundat(?:e|ed|ing|ion)\b',
    r'\bdeluge\b',

    # Standard flood terms
    r'\bflood(?:ed|ing|s)?\b',
    r'\bfloodwater\b',
    r'\bhigh water\b',
    r'\bwater rose\b',
    r'\briver (?:rose|rising|overflow|overflowed)\b',

    # Damage indicators
    r'\bwashed away\b',
    r'\bswept away\b',
    r'\bdam (?:break|broke|burst)\b',
    r'\bice jam\b',
]

FLOOD_PATTERNS = [re.compile(kw, re.IGNORECASE) for kw in HISTORICAL_FLOOD_KEYWORDS]


def has_flood_keywords(text: str) -> dict:
    """Check if text contains flood-related keywords."""
    matches = []
    for pattern in FLOOD_PATTERNS:
        found = pattern.findall(text)
        matches.extend(found)

    return {
        'has_keywords': len(matches) > 0,
        'keyword_count': len(matches),
        'keywords_found': list(set(m.lower() for m in matches))
    }


def extract_year(article: dict) -> int:
    """Extract year from article."""
    date_str = article.get('date', '')
    match = re.search(r'(\d{4})', str(date_str))
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2025:
            return year
    return None


def load_ia_articles(ia_dir: Path, limit: int = None) -> list:
    """Load articles from IA newspapers directory."""
    articles = []
    collections = [d for d in ia_dir.iterdir() if d.is_dir()]

    print(f"Loading from {len(collections)} collections...")

    for collection_dir in collections:
        # Load manifest for metadata
        manifest_file = collection_dir / 'manifest.json'
        metadata_lookup = {}
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            metadata_lookup = {
                r['identifier']: r.get('metadata', {})
                for r in manifest.get('results', [])
            }

        # Find all text files
        for text_file in collection_dir.rglob('*.txt'):
            try:
                try:
                    text = text_file.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    text = text_file.read_text(encoding='latin-1')

                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) < 200:
                    continue
                if len(text) > 4000:
                    text = text[:4000]

                identifier = text_file.parent.name
                metadata = metadata_lookup.get(identifier, {})

                articles.append({
                    'id': f"ia-{identifier}",
                    'title': metadata.get('title', '') or identifier,
                    'full_text': text,
                    'date': metadata.get('date', ''),
                    'collection': collection_dir.name,
                })
            except:
                continue

        if limit and len(articles) >= limit:
            break

    return articles[:limit] if limit else articles


def run_bert_inference(articles: list, model, tokenizer, device, threshold: float) -> list:
    """Run BERT inference."""
    batch_size = 16

    for i in tqdm(range(0, len(articles), batch_size), desc="BERT inference"):
        batch = articles[i:i+batch_size]
        batch_texts = [a['full_text'] for a in batch]

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()

        for article, prob in zip(batch, probs):
            article['bert_probability'] = float(prob)
            article['bert_passes_filter'] = bool(prob > threshold)

    return articles


def main():
    parser = argparse.ArgumentParser(description='Evaluate BERT recall on historical text')
    parser.add_argument('--sample-size', '-n', type=int, default=1000,
                        help='Number of articles to sample')
    parser.add_argument('--output', '-o', type=Path,
                        default=PROJECT_ROOT / 'results' / 'historical_evaluation',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 70)
    print("BERT RECALL EVALUATION ON HISTORICAL TEXT")
    print("=" * 70)
    print(f"Goal: Ensure BERT doesn't filter out actual floods")
    print(f"Sample size: {args.sample_size}")
    print()

    # Load model
    print("1. Loading BERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"   Model: {MODEL_PATH.name}")
    print(f"   Device: {device}")

    # Load threshold
    threshold_file = MODEL_PATH / 'threshold_info.json'
    if threshold_file.exists():
        with open(threshold_file) as f:
            threshold_info = json.load(f)
        threshold = threshold_info['threshold']
    else:
        threshold = 0.30
    print(f"   Threshold: {threshold:.3f}")

    # Load articles
    print("\n2. Loading Internet Archive articles...")
    ia_dir = PROJECT_ROOT / 'data' / 'ia_newspapers'

    if not ia_dir.exists():
        print(f"ERROR: IA directory not found: {ia_dir}")
        return

    all_articles = load_ia_articles(ia_dir, limit=args.sample_size * 2)
    articles = random.sample(all_articles, min(args.sample_size, len(all_articles)))
    print(f"   Sampled {len(articles)} articles")

    # Add keyword flags
    print("\n3. Identifying articles with flood keywords...")
    for article in articles:
        kw = has_flood_keywords(article['full_text'])
        article['has_flood_keywords'] = kw['has_keywords']
        article['keywords_found'] = kw['keywords_found']

    keyword_articles = [a for a in articles if a['has_flood_keywords']]
    print(f"   Found {len(keyword_articles)} articles with flood keywords")

    if not keyword_articles:
        print("\n   WARNING: No flood keywords found in sample.")
        print("   Try increasing sample size or checking data.")
        return

    # Run BERT
    print("\n4. Running BERT inference...")
    articles = run_bert_inference(articles, model, tokenizer, device, threshold)

    # Calculate recall
    print("\n5. Evaluating recall...")

    # KEY METRIC: Of articles with flood keywords, how many does BERT let through?
    keyword_passed = [a for a in keyword_articles if a['bert_passes_filter']]
    keyword_filtered = [a for a in keyword_articles if not a['bert_passes_filter']]

    recall = len(keyword_passed) / len(keyword_articles)

    print(f"\n   {'='*50}")
    print(f"   RECALL ON FLOOD-KEYWORD ARTICLES: {recall:.1%}")
    print(f"   {'='*50}")
    print(f"   Articles with flood keywords: {len(keyword_articles)}")
    print(f"   BERT let through (good):      {len(keyword_passed)}")
    print(f"   BERT filtered out (BAD):      {len(keyword_filtered)}")

    # Break down by historical term
    print("\n   Recall by keyword type:")
    keyword_recall = defaultdict(lambda: {'total': 0, 'passed': 0})
    for article in keyword_articles:
        for kw in article['keywords_found']:
            keyword_recall[kw]['total'] += 1
            if article['bert_passes_filter']:
                keyword_recall[kw]['passed'] += 1

    for kw, stats in sorted(keyword_recall.items(), key=lambda x: -x[1]['total']):
        kw_recall = stats['passed'] / stats['total'] if stats['total'] > 0 else 0
        status = "OK" if kw_recall >= 0.9 else "LOW" if kw_recall >= 0.7 else "BAD"
        print(f"   {kw:<20} {stats['passed']:3d}/{stats['total']:3d} = {kw_recall:5.1%}  [{status}]")

    # Save filtered articles for review (these are potential problems)
    args.output.mkdir(parents=True, exist_ok=True)

    if keyword_filtered:
        output_filtered = args.output / 'filtered_flood_articles_REVIEW.json'
        # Include text snippet for easy review
        for a in keyword_filtered:
            a['text_snippet'] = a['full_text'][:500] + '...'
        with open(output_filtered, 'w') as f:
            json.dump(keyword_filtered, f, indent=2, default=str)
        print(f"\n   REVIEW THESE: {output_filtered}")
        print(f"   These {len(keyword_filtered)} articles have flood keywords but BERT filtered them out.")

    # Summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'sample_size': len(articles),
        'keyword_articles': len(keyword_articles),
        'recall': recall,
        'passed': len(keyword_passed),
        'filtered': len(keyword_filtered),
        'threshold': threshold,
    }

    output_summary = args.output / 'recall_evaluation.json'
    with open(output_summary, 'w') as f:
        json.dump(summary, f, indent=2)

    # Final verdict
    print(f"\n{'='*70}")
    if recall >= 0.90:
        print("RESULT: BERT recall is GOOD for historical text")
        print("Proceed with inference on full IA corpus.")
    elif recall >= 0.75:
        print("RESULT: BERT recall is ACCEPTABLE")
        print("Some historical floods may be missed. Review filtered articles.")
        print("Consider lowering threshold or adding historical training examples.")
    else:
        print("RESULT: BERT recall is TOO LOW for historical text")
        print(f"BERT is filtering out {100-recall*100:.0f}% of flood-keyword articles.")
        print("\nRecommendations:")
        print("  1. Lower the threshold (currently {threshold:.2f})")
        print("  2. Add historical articles to training data")
        print("  3. Review filtered_flood_articles_REVIEW.json")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
