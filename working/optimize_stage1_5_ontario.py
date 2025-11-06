"""
Stage 1.5: Ontario Identification
Checks if flood events occurred in Ontario, Canada (only for flood-positive articles)
"""

import json
import dspy
import pandas

import config
import metrics
from signatures import isOntario
from utils import prepare_data

print("\n" + "="*60)
print("STAGE 1.5: ONTARIO IDENTIFICATION")
print("="*60)

# Load data
with open(config.train_filepath, 'r') as file:
    train_set = prepare_data(json.load(file))

with open(config.test_filepath, 'r') as file:
    test_set = prepare_data(json.load(file))

# Filter for flood-positive articles only
flood_positive_train = [ex for ex in train_set if ex.flood_mentioned == True]
flood_positive_test = [ex for ex in test_set if ex.flood_mentioned == True]

print(f"Flood-positive training examples: {len(flood_positive_train)}")
print(f"Flood-positive test examples: {len(flood_positive_test)}")

# Configure LM
lm = dspy.LM(
    config.MODEL_CONFIG['name'],
    api_base=config.MODEL_CONFIG['api_base'],
    api_key=config.MODEL_CONFIG['api_key']
)
dspy.configure(lm=lm)

# Create predictor and evaluator
ontario_checker = dspy.ChainOfThought(isOntario)

evaluate_ontario = dspy.Evaluate(
    devset=flood_positive_test,
    metric=metrics.ontario_correctness_metric,
    num_threads=1,
    display_progress=True,
    display_table=True
)

# Baseline evaluation
print("\nEvaluating baseline...")
baseline_ontario = evaluate_ontario(ontario_checker)
print(f"Baseline Score: {baseline_ontario.score:.2f}%")

# Optimize
print("\nOptimizing Ontario identification...")
ontario_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=metrics.ontario_correctness_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=15,
    num_threads=4
)

optimized_ontario = ontario_optimizer.compile(
    ontario_checker,
    trainset=flood_positive_train
)

# Evaluate optimized
print("\nEvaluating optimized model...")
optimized_ontario_result = evaluate_ontario(optimized_ontario)
print(f"Optimized Score: {optimized_ontario_result.score:.2f}%")
print(f"Improvement: {optimized_ontario_result.score - baseline_ontario.score:.2f}%")

# Calculate detailed metrics
print("\n" + "="*60)
print("DETAILED METRICS")
print("="*60)

results = optimized_ontario_result.results
tp = sum(1 for ex, pred, _ in results if ex.is_ontario and pred.is_ontario)
fp = sum(1 for ex, pred, _ in results if not ex.is_ontario and pred.is_ontario)
fn = sum(1 for ex, pred, _ in results if ex.is_ontario and not pred.is_ontario)
tn = sum(1 for ex, pred, _ in results if not ex.is_ontario and not pred.is_ontario)

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

print(f"\nConfusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.2%}")

# Save optimized model
optimized_ontario.save('models/stage1_5_optimized.json')
print("\n✓ Model saved to: models/stage1_5_optimized.json")

# Export results
stage1_5_results = []
for example, prediction, score in optimized_ontario_result.results:
    stage1_5_results.append({
        'article_snippet': example.article_text[:150] + '...',
        'publication_date': example.publication_date,
        'expected_ontario': example.is_ontario,
        'predicted_ontario': prediction.is_ontario,
        'correct': score,
        'reasoning': prediction.reasoning
    })

df = pandas.DataFrame(stage1_5_results)
df.to_csv('results/stage1_5_results.csv', index=False)
print(f"✓ Results saved to: results/stage1_5_results.csv ({len(df)} articles)")

print("\n" + "="*60)
print("STAGE 1.5 COMPLETE")
print("="*60)