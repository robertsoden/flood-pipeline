# Python
import json
import subprocess

# External Libraries
import pandas
import dspy

# This App
import config
import metrics
from signatures import floodIdentification,floodDetails,isOntario
from utils import prepare_data

# PX00
import phoenix as px
from phoenix.otel import register

# Launch Phoenix
print("\n" + "="*60)
print("LAUNCHING PHOENIX")
print("="*60)

session = px.launch_app()
print(f"\n🔍 Phoenix UI: {session.url}")
print("Open this URL in your browser to see traces!\n")

# Register auto-instrumentation
register(
    project_name="flood-extraction",
    auto_instrument=True
)

# Load training data
with open(config.train_filepath, 'r') as file:
    train_set = prepare_data(json.load(file))
print(f"Prepared {len(train_set)} training examples")

# Create flood-positive training subset for stage 2 training
flood_positive_train = [ex for ex in train_set if ex.flood_mentioned == True]
print(f"Flood-positive training examples: {len(flood_positive_train)}")

# Load test data
with open(config.test_filepath, 'r') as file:
    test_set = prepare_data(json.load(file))
print(f"Prepared {len(test_set)} test examples")

# Create flood-positive test subset for stage 2 evaluation
flood_positive_test = [ex for ex in test_set if ex.flood_mentioned == True]
print(f"Flood-positive test examples: {len(flood_positive_test)}")

ontario_flood_test = [ex for ex in test_set 
                      if ex.flood_mentioned == True and ex.is_ontario == True]
print(f"Ontario flood test examples: {len(ontario_flood_test)}")

# Create base predictors
flood_extractor = dspy.ChainOfThought(floodIdentification)
ontario_checker = dspy.ChainOfThought(isOntario)
details_extractor = dspy.ChainOfThought(floodDetails)

# Configure language model
lm = dspy.LM(
    config.MODEL_CONFIG['name'],           
    api_base=config.MODEL_CONFIG['api_base'], 
    api_key=config.MODEL_CONFIG['api_key']
)
dspy.configure(lm=lm)

# Create evaluators
evaluate_extraction = dspy.Evaluate(
    devset=test_set,
    metric=metrics.extraction_correctness_metric,
    num_threads=1,
    display_progress=True,
    display_table=True
)

evaluate_details = dspy.Evaluate(
    devset=ontario_flood_test,
    metric=metrics.details_correctness_metric,
    num_threads=1,
    display_progress=True,
    display_table=True
)

print("\n" + "="*60)
print("STAGE 1: OPTIMIZING FLOOD IDENTIFICATION")
print("="*60)

# Baseline for stage 1
print("\nEvaluating baseline flood extractor...")
baseline_extraction = evaluate_extraction(flood_extractor)
print(f"Baseline Extraction Score: {baseline_extraction.score:.2f}%")

# Optimize stage 1 on all data
print("\nOptimizing flood identification...")
extraction_optimizer = dspy.BootstrapFewShotWithRandomSearch(
    metric=metrics.extraction_correctness_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=15,
    num_threads=4
)

optimized_extractor = extraction_optimizer.compile(
    flood_extractor,
    trainset=train_set
)

print("\nEvaluating optimized flood extractor...")
optimized_extraction = evaluate_extraction(optimized_extractor)
print(f"Optimized Extraction Score: {optimized_extraction.score:.2f}%")
print(f"Improvement: {optimized_extraction.score - baseline_extraction.score:.2f}%")


print("\n" + "="*60)
print("STAGE 1.5: OPTIMIZING ONTARIO IDENTIFICATION")
print("="*60)

# Create flood-positive datasets for ontario checking
flood_positive_train_ontario = [ex for ex in train_set if ex.flood_mentioned == True]
flood_positive_test_ontario = [ex for ex in test_set if ex.flood_mentioned == True]

# Create evaluator for ontario checking (you'll need a metric for this)
evaluate_ontario = dspy.Evaluate(
    devset=flood_positive_test_ontario,
    metric=metrics.ontario_correctness_metric,  # You'll need to create this
    num_threads=1,
    display_progress=True,
    display_table=True
)

# Baseline
print("\nEvaluating baseline Ontario checker...")
baseline_ontario = evaluate_ontario(ontario_checker)
print(f"Baseline Ontario Score: {baseline_ontario.score:.2f}%")

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
    trainset=flood_positive_train_ontario
)

print("\nEvaluating optimized Ontario checker...")
optimized_ontario_result = evaluate_ontario(optimized_ontario)
print(f"Optimized Ontario Score: {optimized_ontario_result.score:.2f}%")

# Ceate Ontario-only datasets for Stage 2
ontario_flood_train = [ex for ex in train_set 
                       if ex.flood_mentioned == True and ex.is_ontario == True]
print(f"Ontario flood training examples: {len(ontario_flood_train)}")

# Stage 2: Details Extraction
print("\n" + "="*60)
print("STAGE 2: OPTIMIZING DETAIL EXTRACTION")
print("="*60)

# Baseline for stage 2
print("\nEvaluating baseline details extractor on flood-positive examples...")
baseline_details = evaluate_details(details_extractor)
print(f"Baseline Details Score: {baseline_details.score:.2f}%")

# Optimize stage 2 on flood-positive data only
print("\nOptimizing detail extraction...")
details_optimizer = dspy.MIPROv2(
    metric=metrics.details_correctness_metric,
    auto="medium",
    verbose=True
)

optimized_details = details_optimizer.compile(
    details_extractor,
    trainset=flood_positive_train
)

print("\nEvaluating optimized details extractor...")
optimized_details_result = evaluate_details(optimized_details)
print(f"Optimized Details Score: {optimized_details_result.score:.2f}%")
print(f"Improvement: {optimized_details_result.score - baseline_details.score:.2f}%")
    
# Combine results into single CSV
print("\n" + "="*60)
print("EXPORTING COMBINED RESULTS")
print("="*60)

# Start with Stage 1 results (all articles)
combined_results = {}
for example, prediction, score in optimized_extraction.results:
    combined_results[example.article_text] = {
        'article_snippet': example.article_text[:150] + '...',
        'publication_date': example.publication_date,
        'expected_flood': example.flood_mentioned,
        'predicted_flood': prediction.flood_mentioned,
        'stage1_correct': score,
        'stage1_reasoning': prediction.reasoning,
        # Initialize detail fields as empty
        'expected_ontario': None,   
        'predicted_ontario': None,  
        'stage1.5_correct': None,   
        'expected_location': '',
        'predicted_location': '',
        'expected_date': '',
        'predicted_date': '',
        'expected_impacts': '',
        'predicted_impacts': '',
        'stage2_correct': None,
        'stage2_reasoning': ''
    }

# Stage 1.5 results (is Ontario?)
for example, prediction, score in optimized_ontario_result.results:
    if example.article_text in combined_results:
        combined_results[example.article_text].update({
            'expected_ontario': example.is_ontario,
            'predicted_ontario': prediction.is_ontario,
            'stage1.5_correct': score,
        })

# Add Stage 2 results (flood-positive articles only)
for example, prediction, score in optimized_details_result.results:
    if example.article_text in combined_results:
        # Use stored results - no recalculation, no extra LLM calls!
        field_results = prediction.field_results if hasattr(prediction, 'field_results') else {}
        
        combined_results[example.article_text].update({
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
            'stage2_correct': score,
            'stage2_reasoning': prediction.reasoning
        })

# Convert to DataFrame and save
df = pandas.DataFrame(list(combined_results.values()))
df.to_csv('evaluation_results.csv', index=False)
print(f"Combined results saved: {len(df)} articles → evaluation_results.csv")

# Stage 1 F1 Metrics
print("\n" + "="*60)
print("STAGE 1 DETAILED METRICS")
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

print("\n" + "="*60)
print("OPTIMIZATION COMPLETE")
print("="*60)

print("\n" + "="*60)
print("STAGE 1.5 DETAILED METRICS (ONTARIO IDENTIFICATION)")
print("="*60)

ontario_results = optimized_ontario_result.results
ont_tp = sum(1 for ex, pred, _ in ontario_results if ex.is_ontario and pred.is_ontario)
ont_fp = sum(1 for ex, pred, _ in ontario_results if not ex.is_ontario and pred.is_ontario)
ont_fn = sum(1 for ex, pred, _ in ontario_results if ex.is_ontario and not pred.is_ontario)
ont_tn = sum(1 for ex, pred, _ in ontario_results if not ex.is_ontario and not pred.is_ontario)

ont_precision = ont_tp / (ont_tp + ont_fp) if (ont_tp + ont_fp) > 0 else 0.0
ont_recall = ont_tp / (ont_tp + ont_fn) if (ont_tp + ont_fn) > 0 else 0.0
ont_f1 = 2 * ont_precision * ont_recall / (ont_precision + ont_recall) if (ont_precision + ont_recall) > 0 else 0.0

print(f"\nConfusion Matrix: TP={ont_tp}, FP={ont_fp}, FN={ont_fn}, TN={ont_tn}")
print(f"Precision: {ont_precision:.2%} | Recall: {ont_recall:.2%} | F1: {ont_f1:.2%}")

# Keep Phoenix running
print("\n🔍 Phoenix is still running!")
print(f"   Open: {session.url}")
print("\nPress Ctrl+C to stop Phoenix and exit...")

try:
    while True:
        import time
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\nShutting down Phoenix...")
    print("Goodbye!")