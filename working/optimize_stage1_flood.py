"""
Stage 1: Flood Identification
Detects whether articles mention specific flood events
"""

import json
import dspy
import pandas

import config
import metrics
from signatures import floodIdentification
from utils import prepare_data

print("\n" + "="*60)
print("STAGE 1: FLOOD IDENTIFICATION")
print("="*60)

# Load data
with open(config.train_filepath, 'r') as file:
    train_set = prepare_data(json.load(file))
print(f"Training examples: {len(train_set)}")

with open(config.test_filepath, 'r') as file:
    test_set = prepare_data(json.load(file))
print(f"Test examples: {len(test_set)}")

# Configure LM
lm = dspy.LM(
    config.MODEL_CONFIG['name'],
    api_base=config.MODEL_CONFIG['api_base'],
    api_key=config.MODEL_CONFIG['api_key']
)
dspy.configure(lm=lm)

# Create predictor and evaluator
flood_extractor = dspy.ChainOfThought(floodIdentification)

evaluate_extraction = dspy.Evaluate(
    devset=test_set,
    metric=metrics.extraction_precision_focused_metric,
    num_threads=4,
    display_progress=True,
    display_table=True
)

# Baseline evaluation
print("\nEvaluating baseline...")
baseline_extraction = evaluate_extraction(flood_extractor)
print(f"Baseline Score: {baseline_extraction.score:.2f}%")

# Optimize
print("\nOptimizing flood identification...")
extraction_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=metrics.extraction_precision_focused_metric,  
    max_bootstrapped_demos=6,    
    max_labeled_demos=6,      
    num_candidate_programs=25,
    num_threads=4
)

optimized_extractor = extraction_optimizer.compile(
    flood_extractor,
    trainset=train_set
)

# Evaluate optimized
print("\nEvaluating optimized model...")
optimized_extraction = evaluate_extraction(optimized_extractor)
print(f"Optimized Score: {optimized_extraction.score:.2f}%")
print(f"Improvement: {optimized_extraction.score - baseline_extraction.score:.2f}%")

# Calculate detailed metrics
print("\n" + "="*60)
print("DETAILED METRICS")
print("="*60)

results = optimized_extraction.results
tp = sum(1 for ex, pred, _ in results if ex.flood_mentioned and pred.flood_mentioned)
fp = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and pred.flood_mentioned)
fn = sum(1 for ex, pred, _ in results if ex.flood_mentioned and not pred.flood_mentioned)
tn = sum(1 for ex, pred, _ in results if not ex.flood_mentioned and not pred.flood_mentioned)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nConfusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")

# Save optimized model
optimized_extractor.save('models/stage1_optimized.json')
print("\n✓ Model saved to: models/stage1_optimized.json")

# Export results
stage1_results = []
for example, prediction, score in optimized_extraction.results:
    stage1_results.append({
        'article_snippet': example.article_text[:150] + '...',
        'publication_date': example.publication_date,
        'expected_flood': example.flood_mentioned,
        'predicted_flood': prediction.flood_mentioned,
        'correct': score,
        'reasoning': prediction.reasoning
    })

df = pandas.DataFrame(stage1_results)
df.to_csv('results/stage1_results.csv', index=False)
print(f"✓ Results saved to: results/stage1_results.csv ({len(df)} articles)")