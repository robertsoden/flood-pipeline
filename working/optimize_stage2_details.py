"""
Stage 2: Detail Extraction
Extracts location, date, and impacts for Ontario flood events
"""

import json
import dspy
import pandas

import config
import metrics
from signatures import floodDetails
from utils import prepare_data

# Phoenix setup
import phoenix as px
from phoenix.otel import register

print("\n" + "="*60)
print("STAGE 2: DETAIL EXTRACTION")
print("="*60)

# Load data
with open(config.train_filepath, 'r') as file:
    train_set = prepare_data(json.load(file))

with open(config.test_filepath, 'r') as file:
    test_set = prepare_data(json.load(file))

# Filter for Ontario flood articles only
ontario_flood_train = [ex for ex in train_set 
                       if ex.flood_mentioned == True and ex.is_ontario == True]
ontario_flood_test = [ex for ex in test_set 
                      if ex.flood_mentioned == True and ex.is_ontario == True]

print(f"Ontario flood training examples: {len(ontario_flood_train)}")
print(f"Ontario flood test examples: {len(ontario_flood_test)}")

# Configure LM
lm = dspy.LM(
    config.MODEL_CONFIG['name'],
    api_base=config.MODEL_CONFIG['api_base'],
    api_key=config.MODEL_CONFIG['api_key']
)
dspy.configure(lm=lm)

# Create predictor and evaluator
details_extractor = dspy.ChainOfThought(floodDetails)

evaluate_details = dspy.Evaluate(
    devset=ontario_flood_test,
    metric=metrics.details_correctness_metric,
    num_threads=1,
    display_progress=True,
    display_table=True
)

# Baseline evaluation
print("\nEvaluating baseline...")
baseline_details = evaluate_details(details_extractor)
print(f"Baseline Score: {baseline_details.score:.2f}%")

# Optimize
print("\nOptimizing detail extraction...")
details_optimizer = dspy.MIPROv2(
    metric=metrics.details_correctness_metric,
    auto="medium",
    verbose=True
)

# Note: Using flood_positive_train (all floods) rather than ontario_flood_train
# This gives more training data. Adjust if you want Ontario-specific training.
with open(config.train_filepath, 'r') as file:
    train_set_full = prepare_data(json.load(file))
flood_positive_train = [ex for ex in train_set_full if ex.flood_mentioned == True]

optimized_details = details_optimizer.compile(
    details_extractor,
    trainset=flood_positive_train
)

# Evaluate optimized
print("\nEvaluating optimized model...")
optimized_details_result = evaluate_details(optimized_details)
print(f"Optimized Score: {optimized_details_result.score:.2f}%")
print(f"Improvement: {optimized_details_result.score - baseline_details.score:.2f}%")

# Calculate field-level accuracy
print("\n" + "="*60)
print("FIELD-LEVEL ACCURACY")
print("="*60)

location_correct = 0
date_correct = 0
impacts_correct = 0
reasoning_correct = 0
total = len(optimized_details_result.results)

for example, prediction, score in optimized_details_result.results:
    if hasattr(prediction, 'field_results'):
        field_results = prediction.field_results
        location_correct += field_results.get('location_correct', False)
        date_correct += field_results.get('date_correct', False)
        impacts_correct += field_results.get('impacts_correct', False)
        reasoning_correct += field_results.get('reasoning_correct', False)

print(f"\nLocation Accuracy: {location_correct}/{total} ({location_correct/total:.1%})")
print(f"Date Accuracy: {date_correct}/{total} ({date_correct/total:.1%})")
print(f"Impacts Accuracy: {impacts_correct}/{total} ({impacts_correct/total:.1%})")
print(f"Reasoning Provided: {reasoning_correct}/{total} ({reasoning_correct/total:.1%})")

# Save optimized model
optimized_details.save('models/stage2_optimized.json')
print("\n✓ Model saved to: models/stage2_optimized.json")

# Export results
stage2_results = []
for example, prediction, score in optimized_details_result.results:
    field_results = prediction.field_results if hasattr(prediction, 'field_results') else {}
    
    stage2_results.append({
        'article_snippet': example.article_text[:150] + '...',
        'publication_date': example.publication_date,
        
        'expected_location': example.location,
        'predicted_location': prediction.location,
        'location_correct': field_results.get('location_correct', False),
        
        'expected_date': example.flood_date,
        'predicted_date': prediction.flood_date,
        'date_correct': field_results.get('date_correct', False),
        
        'expected_impacts': example.impacts,
        'predicted_impacts': prediction.impacts,
        'impacts_correct': field_results.get('impacts_correct', False),
        
        'reasoning_correct': field_results.get('reasoning_correct', False),
        'overall_correct': score,
        'reasoning': prediction.reasoning
    })

df = pandas.DataFrame(stage2_results)
df.to_csv('results/stage2_results.csv', index=False)
print(f"✓ Results saved to: results/stage2_results.csv ({len(df)} articles)")