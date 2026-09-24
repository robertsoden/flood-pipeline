"""
High-Recall BERT Training - Using Shared Config
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import dspy

# Import from shared config
from shared.config import (
    train_filepath,
    test_filepath,
    unlabeled_filepath,
    STAGE1_CONFIG,
    PROJECT_ROOT,
    get_config_value,
)
from shared.utils import prepare_data
from shared.logging_config import setup_logger, log_section, log_config

# Setup logging
logger = setup_logger(__name__, 'stage1_train', PROJECT_ROOT)

log_section(logger, "HIGH-RECALL BERT TRAINING (BALANCED DATASET)")

# Configuration from shared config
MODEL_NAME = STAGE1_CONFIG['bert_model']
TARGET_RECALL = STAGE1_CONFIG['target_recall']
BATCH_SIZE = STAGE1_CONFIG['batch_size']
MAX_LENGTH = STAGE1_CONFIG['max_length']
NUM_EPOCHS = STAGE1_CONFIG['num_epochs']
LEARNING_RATE = STAGE1_CONFIG['learning_rate']

# Training-specific settings from config
PSEUDO_LABEL_CONFIDENCE = get_config_value('pseudo_label_confidence', STAGE1_CONFIG)
MAX_PSEUDO_LABELS_PER_ITERATION = get_config_value('max_pseudo_labels_per_iteration', STAGE1_CONFIG)
NUM_ITERATIONS = get_config_value('num_iterations', STAGE1_CONFIG)
RECALL_WEIGHT_MULTIPLIER = get_config_value('recall_weight_multiplier', STAGE1_CONFIG)
VALIDATION_SPLIT = 0.2  # Hold out 20% of training data for validation

config_display = {
    'Model': MODEL_NAME,
    'Target recall': f"{TARGET_RECALL:.0%}",
    'Batch size': BATCH_SIZE,
    'Max length': MAX_LENGTH,
    'Num epochs': NUM_EPOCHS,
    'Learning rate': LEARNING_RATE,
    'Weight multiplier': f"{RECALL_WEIGHT_MULTIPLIER}x",
    'Pseudo-label confidence': PSEUDO_LABEL_CONFIDENCE,
    'Validation split': f"{VALIDATION_SPLIT:.0%}",
}
log_config(logger, config_display, "Configuration")

# Create output directories
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# WEIGHTED TRAINER
# ============================================================================

class BalancedHighRecallTrainer(Trainer):
    """Trainer with weighted loss for high recall"""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            print(f"    ✓ Weights: non-flood={class_weights[0]:.3f}, flood={class_weights[1]:.3f}")
            print(f"    ✓ Flood weight is {class_weights[1]/class_weights[0]:.2f}x higher")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def calculate_balanced_class_weights(train_examples, multiplier=1.1):
    """Calculate class weights for balanced dataset"""
    num_floods = sum(1 for ex in train_examples if ex.flood_mentioned)
    num_non_floods = len(train_examples) - num_floods
    total = len(train_examples)
    
    # Base inverse frequency weighting
    weight_non_flood = total / (2 * num_non_floods)
    weight_flood = total / (2 * num_floods)
    
    # Apply multiplier
    weight_flood *= multiplier
    
    # Normalize
    total_weight = weight_non_flood + weight_flood
    weight_non_flood = weight_non_flood / total_weight * 2
    weight_flood = weight_flood / total_weight * 2
    
    weights = torch.tensor([weight_non_flood, weight_flood], dtype=torch.float32)
    return weights, num_floods, num_non_floods


# ============================================================================
# METRICS
# ============================================================================

def compute_recall_focused_metrics(eval_pred):
    """Calculate metrics with emphasis on recall"""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    # Recall at different thresholds
    recall_at_35 = recall_score(labels, (probs > 0.35).astype(int))
    recall_at_30 = recall_score(labels, (probs > 0.30).astype(int))
    recall_at_25 = recall_score(labels, (probs > 0.25).astype(int))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'recall_at_0.35': recall_at_35,
        'recall_at_0.30': recall_at_30,
        'recall_at_0.25': recall_at_25,
    }


def find_threshold_for_target_recall(probs, labels, target_recall=0.95, min_threshold=None):
    """Find threshold that achieves target recall"""
    if min_threshold is None:
        min_threshold = get_config_value('threshold_min', STAGE1_CONFIG)

    threshold_max = get_config_value('threshold_max', STAGE1_CONFIG)
    threshold_step = get_config_value('threshold_step', STAGE1_CONFIG)

    best_threshold = None
    best_precision = 0

    for threshold in np.arange(threshold_max, min_threshold, -threshold_step):
        pred_labels = (probs > threshold).astype(int)
        
        tp = np.sum((labels == 1) & (pred_labels == 1))
        fn = np.sum((labels == 1) & (pred_labels == 0))
        tn = np.sum((labels == 0) & (pred_labels == 0))
        fp = np.sum((labels == 0) & (pred_labels == 1))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        filter_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if recall >= target_recall:
            if precision > best_precision or best_threshold is None:
                best_threshold = threshold
                best_precision = precision
                best_recall = recall
                best_filter_rate = filter_rate
    
    if best_threshold is None:
        best_threshold = min_threshold
        pred_labels = (probs > best_threshold).astype(int)
        tp = np.sum((labels == 1) & (pred_labels == 1))
        fn = np.sum((labels == 1) & (pred_labels == 0))
        tn = np.sum((labels == 0) & (pred_labels == 0))
        fp = np.sum((labels == 0) & (pred_labels == 1))
        
        best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        best_filter_rate = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return best_threshold, best_recall, best_precision, best_filter_rate


# ============================================================================
# DATA LOADING
# ============================================================================

print("\n1. Loading labeled data...")
with open(train_filepath, 'r') as file:
    labeled_all = prepare_data(json.load(file))
print(f"   Labeled data: {len(labeled_all)} examples")

with open(test_filepath, 'r') as file:
    test_data = prepare_data(json.load(file))
print(f"   Test (held out): {len(test_data)} examples")

# Split labeled data into train and validation sets
# Stratify by flood_mentioned to maintain class balance
labels_for_split = [1 if ex.flood_mentioned else 0 for ex in labeled_all]
labeled_train, validation_data = train_test_split(
    labeled_all,
    test_size=VALIDATION_SPLIT,
    random_state=42,
    stratify=labels_for_split
)
print(f"   Training: {len(labeled_train)} examples")
print(f"   Validation: {len(validation_data)} examples")

# Class distribution
train_floods = sum(1 for ex in labeled_train if ex.flood_mentioned)
train_non_floods = len(labeled_train) - train_floods
val_floods = sum(1 for ex in validation_data if ex.flood_mentioned)
val_non_floods = len(validation_data) - val_floods
print(f"\n   Training distribution:")
print(f"     Floods: {train_floods} ({train_floods/len(labeled_train):.1%})")
print(f"     Non-floods: {train_non_floods} ({train_non_floods/len(labeled_train):.1%})")
print(f"\n   Validation distribution:")
print(f"     Floods: {val_floods} ({val_floods/len(validation_data):.1%})")
print(f"     Non-floods: {val_non_floods} ({val_non_floods/len(validation_data):.1%})")

print("\n2. Loading unlabeled data...")
with open(unlabeled_filepath, 'r') as file:
    unlabeled_raw = json.load(file)
print(f"   Unlabeled: {len(unlabeled_raw):,} articles")

unlabeled_pool = [
    {'article_text': article['full_text'], 
     'publication_date': article.get('publication_date', '')}
    for article in unlabeled_raw
]


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def prepare_dataset(examples):
    """Convert to Hugging Face Dataset format"""
    return Dataset.from_dict({
        'text': [ex.article_text for ex in examples],
        'label': [1 if ex.flood_mentioned else 0 for ex in examples]
    })


def tokenize_dataset(dataset, tokenizer):
    """Tokenize dataset"""
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=MAX_LENGTH
        )
    return dataset.map(tokenize_function, batched=True)


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_high_recall_model(train_examples, val_examples, output_dir, iteration=0):
    """Train BERT model optimized for high recall.

    Uses validation set for model selection and threshold tuning.
    Test set evaluation happens separately at the end of all training.
    """
    print(f"\n{'='*70}")
    print(f"HIGH-RECALL TRAINING - ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")
    
    # Calculate class weights
    class_weights, num_floods, num_non_floods = calculate_balanced_class_weights(
        train_examples, multiplier=RECALL_WEIGHT_MULTIPLIER
    )
    
    print(f"  Class distribution:")
    print(f"    Floods: {num_floods} ({num_floods/len(train_examples):.1%})")
    print(f"    Non-floods: {num_non_floods} ({num_non_floods/len(train_examples):.1%})")
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, problem_type="single_label_classification"
    )
    
    # Prepare datasets
    train_dataset = tokenize_dataset(prepare_dataset(train_examples), tokenizer)
    val_dataset = tokenize_dataset(prepare_dataset(val_examples), tokenizer)
    
    # Training arguments
    # CRITICAL: Use F1 to select best epoch (not recall which would pick epoch 1)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=LEARNING_RATE,
        logging_dir=str(PROJECT_ROOT / 'logs'),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  # Use F1, not recall!
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_SEED,  # For reproducibility
    )
    
    # Train with validation set for model selection
    trainer = BalancedHighRecallTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Use validation, NOT test
        compute_metrics=compute_recall_focused_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )

    print("\n🚀 Starting training...")
    trainer.train()

    # Evaluate on validation set (for threshold tuning)
    val_results = trainer.evaluate(val_dataset)
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:  {val_results['eval_accuracy']:.3f}")
    print(f"  Precision: {val_results['eval_precision']:.3f}")
    print(f"  Recall:    {val_results['eval_recall']:.3f}")
    print(f"  F1:        {val_results['eval_f1']:.3f}")

    # Find optimal threshold using validation set
    predictions = trainer.predict(val_dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    true_labels = predictions.label_ids
    
    threshold, recall, precision, filter_rate = find_threshold_for_target_recall(
        probs, true_labels, target_recall=TARGET_RECALL
    )
    
    print(f"\n🎯 OPTIMAL THRESHOLD FOR {TARGET_RECALL:.0%}+ RECALL:")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Recall:    {recall:.1%} {'✅' if recall >= TARGET_RECALL else '⚠️'}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Filter rate: {filter_rate:.1%}")
    
    # Sample predictions from validation set
    print(f"\n📊 Sample predictions (validation set):")
    model.eval()
    sample_indices = np.random.choice(len(val_examples), min(10, len(val_examples)), replace=False)

    for idx in sample_indices:
        ex = val_examples[idx]
        with torch.no_grad():
            inputs = tokenizer(ex.article_text, return_tensors='pt', truncation=True, max_length=MAX_LENGTH)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[0, 1].item()
            pred = "FLOOD" if prob > threshold else "NOT FLOOD"
        
        actual = "FLOOD" if ex.flood_mentioned else "NOT FLOOD"
        correct = "✓" if (pred == actual) else "✗"
        print(f"  {correct} Actual={actual:10s} Pred={pred:10s} (prob={prob:.3f})")
    
    return model, tokenizer, threshold, recall, precision, filter_rate


# ============================================================================
# PSEUDO-LABELING
# ============================================================================

def generate_pseudo_labels(model, tokenizer, unlabeled_pool, confidence_threshold):
    """Generate high-confidence pseudo-labels"""
    print(f"\n{'='*70}")
    print("GENERATING PSEUDO-LABELS")
    print(f"{'='*70}")
    print(f"Unlabeled pool: {len(unlabeled_pool):,}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    texts = [article['article_text'] for article in unlabeled_pool]
    
    model.eval()
    device = model.device
    all_probs = []
    
    batch_progress_interval = get_config_value('batch_progress_interval', STAGE1_CONFIG)

    print("Predicting on unlabeled data...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts, return_tensors='pt', truncation=True,
                max_length=MAX_LENGTH, padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

        if (i + BATCH_SIZE) % batch_progress_interval == 0:
            print(f"  Processed {min(i+BATCH_SIZE, len(texts)):,}/{len(texts):,}")
    
    print(f"✓ Complete!")
    
    # Select high-confidence predictions
    high_conf_floods = [(idx, prob) for idx, prob in enumerate(all_probs) if prob > confidence_threshold]
    high_conf_non_floods = [(idx, prob) for idx, prob in enumerate(all_probs) if prob < (1 - confidence_threshold)]
    
    print(f"\nHigh-confidence predictions:")
    print(f"  Floods: {len(high_conf_floods)} (prob > {confidence_threshold})")
    print(f"  Non-floods: {len(high_conf_non_floods)} (prob < {1-confidence_threshold})")
    
    # Balance classes
    max_per_class = min(
        MAX_PSEUDO_LABELS_PER_ITERATION // 2,
        len(high_conf_floods),
        len(high_conf_non_floods)
    )
    
    pseudo_labeled = []
    
    # Add floods
    selected_floods = sorted(high_conf_floods, key=lambda x: x[1], reverse=True)[:max_per_class]
    for idx, prob in selected_floods:
        article = unlabeled_pool[idx]
        pseudo_labeled.append(
            dspy.Example(
                article_text=article['article_text'],
                publication_date=article['publication_date'],
                flood_mentioned=True
            ).with_inputs('article_text')
        )
    
    # Add non-floods
    selected_non_floods = sorted(high_conf_non_floods, key=lambda x: x[1])[:max_per_class]
    for idx, prob in selected_non_floods:
        article = unlabeled_pool[idx]
        pseudo_labeled.append(
            dspy.Example(
                article_text=article['article_text'],
                publication_date=article['publication_date'],
                flood_mentioned=False
            ).with_inputs('article_text')
        )
    
    print(f"\nSelected: {len(selected_floods)} floods, {len(selected_non_floods)} non-floods")
    
    # Remove used articles
    used_indices = set([idx for idx, _ in selected_floods] + [idx for idx, _ in selected_non_floods])
    unlabeled_pool = [article for idx, article in enumerate(unlabeled_pool) if idx not in used_indices]
    
    return pseudo_labeled, unlabeled_pool


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "="*70)
print("STARTING TRAINING LOOP")
print("="*70)

current_train_set = labeled_train.copy()
remaining_unlabeled = unlabeled_pool.copy()
iteration_results = []

for iteration in range(NUM_ITERATIONS):
    # Train model using validation set for selection (NOT test set)
    output_dir = MODELS_DIR / f'balanced_high_recall_iter{iteration}'
    model, tokenizer, threshold, recall, precision, filter_rate = train_high_recall_model(
        current_train_set, validation_data, output_dir, iteration
    )
    
    iteration_results.append({
        'iteration': iteration,
        'train_size': len(current_train_set),
        'threshold': threshold,
        'recall': recall,           # validation recall
        'precision': precision,     # validation precision
        'filter_rate': filter_rate, # validation filter rate
    })
    
    # Generate pseudo-labels
    if iteration < NUM_ITERATIONS - 1:
        pseudo_labeled, remaining_unlabeled = generate_pseudo_labels(
            model, tokenizer, remaining_unlabeled, PSEUDO_LABEL_CONFIDENCE
        )
        
        if len(pseudo_labeled) == 0:
            print("\n⚠️  No pseudo-labels generated. Stopping.")
            break
        
        current_train_set.extend(pseudo_labeled)
        print(f"\nUpdated training set: {len(current_train_set)} examples")
    
    # Save model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    with open(output_dir / 'threshold_info.json', 'w') as f:
        json.dump({
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'filter_rate': filter_rate,
            'train_size': len(current_train_set),
        }, f, indent=2)
    
    print(f"✓ Saved to: {output_dir}")


# ============================================================================
# FINAL TEST SET EVALUATION (UNBIASED)
# ============================================================================

print("\n" + "="*70)
print("FINAL TEST SET EVALUATION (UNBIASED)")
print("="*70)
print("\nThis evaluation uses the held-out test set that was NOT used for")
print("model selection or threshold tuning, providing unbiased metrics.\n")

# Load best model for final evaluation
best_iter_for_test = max(range(NUM_ITERATIONS), key=lambda i: iteration_results[i]['recall'] if iteration_results[i]['recall'] >= TARGET_RECALL else -1)
if iteration_results[best_iter_for_test]['recall'] < TARGET_RECALL:
    best_iter_for_test = max(range(NUM_ITERATIONS), key=lambda i: iteration_results[i]['recall'])

best_model_dir = MODELS_DIR / f'balanced_high_recall_iter{best_iter_for_test}'
best_threshold = iteration_results[best_iter_for_test]['threshold']

print(f"Loading best model from iteration {best_iter_for_test}...")
test_tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
test_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))
test_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
test_model.eval()

# Predict on test set
test_texts = [ex.article_text for ex in test_data]
test_labels = np.array([1 if ex.flood_mentioned else 0 for ex in test_data])
test_probs = []

print(f"Evaluating on {len(test_data)} held-out test examples...")
for i in range(0, len(test_texts), BATCH_SIZE):
    batch_texts = test_texts[i:i+BATCH_SIZE]
    with torch.no_grad():
        inputs = test_tokenizer(
            batch_texts, return_tensors='pt', truncation=True,
            max_length=MAX_LENGTH, padding=True
        )
        inputs = {k: v.to(test_model.device) for k, v in inputs.items()}
        outputs = test_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
        test_probs.extend(probs)

test_probs = np.array(test_probs)
test_preds = (test_probs > best_threshold).astype(int)

# Calculate unbiased metrics
tp = np.sum((test_labels == 1) & (test_preds == 1))
fn = np.sum((test_labels == 1) & (test_preds == 0))
tn = np.sum((test_labels == 0) & (test_preds == 0))
fp = np.sum((test_labels == 0) & (test_preds == 1))

test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0
test_filter_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\n{'='*70}")
print("UNBIASED TEST SET METRICS")
print(f"{'='*70}")
print(f"  Threshold (from validation): {best_threshold:.3f}")
print(f"  Recall:      {test_recall:.1%} {'✅' if test_recall >= TARGET_RECALL else '⚠️'}")
print(f"  Precision:   {test_precision:.1%}")
print(f"  F1:          {test_f1:.3f}")
print(f"  Filter rate: {test_filter_rate:.1%}")
print(f"\n  Confusion Matrix:")
print(f"    TP={tp}, FN={fn}")
print(f"    FP={fp}, TN={tn}")

# Compare validation vs test metrics
val_recall = iteration_results[best_iter_for_test]['recall']
val_precision = iteration_results[best_iter_for_test]['precision']
print(f"\n  Validation vs Test Comparison:")
print(f"    Recall:    {val_recall:.1%} (val) vs {test_recall:.1%} (test) - diff: {(test_recall - val_recall)*100:+.1f}pp")
print(f"    Precision: {val_precision:.1%} (val) vs {test_precision:.1%} (test) - diff: {(test_precision - val_precision)*100:+.1f}pp")

if abs(test_recall - val_recall) > 0.05:
    print(f"\n  ⚠️  Warning: >5pp difference between validation and test recall.")
    print(f"     This may indicate overfitting to validation set.")

# Save test results
test_results_file = RESULTS_DIR / 'test_set_evaluation.json'
with open(test_results_file, 'w') as f:
    json.dump({
        'model_iteration': best_iter_for_test,
        'threshold': best_threshold,
        'test_recall': test_recall,
        'test_precision': test_precision,
        'test_f1': test_f1,
        'test_filter_rate': test_filter_rate,
        'validation_recall': val_recall,
        'validation_precision': val_precision,
        'confusion_matrix': {'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn)},
        'test_set_size': len(test_data),
    }, f, indent=2)
print(f"\n✓ Test results saved: {test_results_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

df_iterations = pd.DataFrame(iteration_results)
print("\nIteration Summary (validation metrics):")
print(df_iterations.to_string(index=False))

# Find best iteration: Among models meeting recall target, pick highest precision
meeting_target = df_iterations[df_iterations['recall'] >= TARGET_RECALL]

if len(meeting_target) > 0:
    best_iter_idx = meeting_target['precision'].idxmax()
    best_iter = df_iterations.loc[best_iter_idx]
    print(f"\n✅ {len(meeting_target)} iteration(s) met {TARGET_RECALL:.0%}+ recall target")
    print(f"   Selected iteration {int(best_iter['iteration'])} with highest precision ({best_iter['precision']:.1%})")
else:
    best_iter_idx = df_iterations['recall'].idxmax()
    best_iter = df_iterations.loc[best_iter_idx]
    print(f"\n⚠️  No iteration met {TARGET_RECALL:.0%}+ recall target")
    print(f"   Selected iteration {int(best_iter['iteration'])} with highest recall ({best_iter['recall']:.1%})")

print(f"\n🏆 BEST ITERATION: {int(best_iter['iteration'])}")
print(f"  Recall:      {best_iter['recall']:.1%} {'✅' if best_iter['recall'] >= TARGET_RECALL else '⚠️'}")
print(f"  Precision:   {best_iter['precision']:.1%}")
print(f"  Threshold:   {best_iter['threshold']:.3f}")
print(f"  Filter rate: {best_iter['filter_rate']:.1%}")

# Save results
results_file = RESULTS_DIR / 'balanced_high_recall_iterations.csv'
df_iterations.to_csv(results_file, index=False)
print(f"\n✓ Results saved: {results_file}")

print("\n" + "="*70)
print("USAGE")
print("="*70)
print(f"\nBest model: models/balanced_high_recall_iter{int(best_iter['iteration'])}")
print(f"Threshold: {best_iter['threshold']:.3f}")
print(f"\nUnbiased test set performance:")
print(f"  Recall:    {test_recall:.1%}")
print(f"  Precision: {test_precision:.1%}")
print("\n" + "="*70 + "\n")