"""
High-Recall BERT Training - Using Shared Config
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
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
)

print("\n" + "="*70)
print("HIGH-RECALL BERT TRAINING (BALANCED DATASET)")
print("="*70)

# Configuration from shared config
MODEL_NAME = STAGE1_CONFIG.get('bert_model', 'distilbert-base-uncased')
TARGET_RECALL = STAGE1_CONFIG.get('target_recall', 0.95)
BATCH_SIZE = STAGE1_CONFIG.get('batch_size', 16)
MAX_LENGTH = STAGE1_CONFIG.get('max_length', 512)
NUM_EPOCHS = STAGE1_CONFIG.get('num_epochs', 4)
LEARNING_RATE = STAGE1_CONFIG.get('learning_rate', 2e-5)

# Training-specific settings
PSEUDO_LABEL_CONFIDENCE = 0.95
MAX_PSEUDO_LABELS_PER_ITERATION = 3000
NUM_ITERATIONS = 3
RECALL_WEIGHT_MULTIPLIER = 1.1  # Conservative for balanced data

print("Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Target recall: {TARGET_RECALL:.0%}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max length: {MAX_LENGTH}")
print(f"  Num epochs: {NUM_EPOCHS}")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Weight multiplier: {RECALL_WEIGHT_MULTIPLIER}x (conservative)")
print(f"  Pseudo-label confidence: {PSEUDO_LABEL_CONFIDENCE}")

# Create output directories
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_data(raw_data):
    """Convert raw JSON to DSPy Examples"""
    examples = []
    for item in raw_data:
        ex = dspy.Example(
            article_text=item.get('full_text', item.get('article_text', '')),
            publication_date=item.get('publication_date', ''),
            flood_mentioned=item.get('flood_mentioned', False)
        ).with_inputs('article_text')
        examples.append(ex)
    return examples


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


def find_threshold_for_target_recall(probs, labels, target_recall=0.95, min_threshold=0.10):
    """Find threshold that achieves target recall"""
    best_threshold = None
    best_precision = 0
    
    for threshold in np.arange(0.50, min_threshold, -0.01):
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
    labeled_train = prepare_data(json.load(file))
print(f"   Training: {len(labeled_train)} examples")

with open(test_filepath, 'r') as file:
    test_data = prepare_data(json.load(file))
print(f"   Test: {len(test_data)} examples")

# Class distribution
train_floods = sum(1 for ex in labeled_train if ex.flood_mentioned)
train_non_floods = len(labeled_train) - train_floods
print(f"\n   Training distribution:")
print(f"     Floods: {train_floods} ({train_floods/len(labeled_train):.1%})")
print(f"     Non-floods: {train_non_floods} ({train_non_floods/len(labeled_train):.1%})")

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

def train_high_recall_model(train_examples, test_examples, output_dir, iteration=0):
    """Train BERT model optimized for high recall"""
    print(f"\n{'='*70}")
    print(f"HIGH-RECALL TRAINING - ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Training examples: {len(train_examples)}")
    
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
    test_dataset = tokenize_dataset(prepare_dataset(test_examples), tokenizer)
    
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
        save_total_limit=2
    )
    
    # Train
    trainer = BalancedHighRecallTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_recall_focused_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Evaluate
    test_results = trainer.evaluate(test_dataset)
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.3f}")
    print(f"  Precision: {test_results['eval_precision']:.3f}")
    print(f"  Recall:    {test_results['eval_recall']:.3f}")
    print(f"  F1:        {test_results['eval_f1']:.3f}")
    
    # Find optimal threshold
    predictions = trainer.predict(test_dataset)
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
    
    # Sample predictions
    print(f"\n📊 Sample predictions:")
    model.eval()
    sample_indices = np.random.choice(len(test_examples), min(10, len(test_examples)), replace=False)
    
    for idx in sample_indices:
        ex = test_examples[idx]
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
        
        if (i + BATCH_SIZE) % 4000 == 0:
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
    # Train model
    output_dir = MODELS_DIR / f'balanced_high_recall_iter{iteration}'
    model, tokenizer, threshold, recall, precision, filter_rate = train_high_recall_model(
        current_train_set, test_data, output_dir, iteration
    )
    
    iteration_results.append({
        'iteration': iteration,
        'train_size': len(current_train_set),
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'filter_rate': filter_rate,
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
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)

df_iterations = pd.DataFrame(iteration_results)
print("\nIteration Summary:")
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
print("\n" + "="*70 + "\n")