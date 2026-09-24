"""
High-Recall BERT Training with Semi-Supervised Pseudo-Labeling

A standalone, adaptable script for training BERT classifiers optimized for
high recall (catching most positive cases) using semi-supervised learning.

APPROACH:
---------
1. Train on labeled data with weighted loss (higher weight on positive class)
2. Use high-confidence predictions on unlabeled data as pseudo-labels
3. Iterate to expand training set
4. Find optimal threshold to achieve target recall

KEY FEATURES:
-------------
- Proper train/validation/test split (no data leakage)
- Weighted loss for recall optimization
- Semi-supervised pseudo-labeling
- Threshold tuning for target recall
- Reproducible with fixed random seeds

ADAPTATION GUIDE:
-----------------
1. Update CONFIG section with your paths and parameters
2. Update prepare_data() to match your data format
3. Adjust CLASS_WEIGHTS_MULTIPLIER if needed (higher = more recall, less precision)
4. Set TARGET_RECALL to your desired minimum recall

REQUIREMENTS:
-------------
pip install torch transformers datasets scikit-learn pandas numpy tqdm

Author: Adapted from flood_news_pipeline project
"""

import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset
import dspy  # Only needed if using DSPy Example format; can be removed

# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION FOR YOUR PROJECT
# =============================================================================

CONFIG = {
    # Paths - UPDATE THESE
    'train_data_path': 'data/train.json',      # Labeled training data
    'test_data_path': 'data/test.json',        # Held-out test data
    'unlabeled_data_path': 'data/unlabeled.json',  # Unlabeled data for pseudo-labeling
    'output_dir': 'models',                     # Where to save trained models
    'results_dir': 'results',                   # Where to save results

    # Model settings
    'bert_model': 'distilbert-base-uncased',   # Base model (can use bert-base-uncased, roberta-base, etc.)
    'max_length': 512,                          # Max tokens (reduce if memory issues)
    'batch_size': 16,                           # Batch size (reduce if memory issues)

    # Training settings
    'num_epochs': 4,                            # Max epochs per iteration
    'learning_rate': 2e-5,                      # Learning rate
    'validation_split': 0.2,                    # Fraction of train data for validation

    # High-recall optimization
    'target_recall': 0.95,                      # Target recall (0.95 = catch 95% of positives)
    'class_weight_multiplier': 1.1,             # Positive class weight boost (higher = more recall)

    # Pseudo-labeling settings
    'num_iterations': 3,                        # Number of self-training iterations
    'pseudo_label_confidence': 0.95,            # Min confidence for pseudo-labels
    'max_pseudo_labels_per_iteration': 3000,    # Max pseudo-labels to add per iteration

    # Threshold search
    'threshold_min': 0.10,                      # Minimum threshold to consider
    'threshold_max': 0.50,                      # Maximum threshold to consider
    'threshold_step': 0.005,                    # Step size for threshold search
}

# Random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =============================================================================
# DATA LOADING - MODIFY THIS FOR YOUR DATA FORMAT
# =============================================================================

def prepare_data(raw_data: list) -> list:
    """
    Convert raw JSON data to training examples.

    MODIFY THIS FUNCTION to match your data format.

    Expected output: List of objects with:
        - article_text: str (the text to classify)
        - flood_mentioned: bool (True = positive class, False = negative)

    Current format expects JSON like:
        {"full_text": "...", "annotations": {"flood_mentioned": true}}
    or:
        {"article_text": "...", "flood_mentioned": true}
    """
    examples = []

    for item in raw_data:
        # Handle nested annotations format
        if 'annotations' in item:
            article_text = item.get('full_text', '')
            label = item['annotations'].get('flood_mentioned', False)
        # Handle flat format
        else:
            article_text = item.get('article_text', item.get('full_text', ''))
            label = item.get('flood_mentioned', item.get('label', False))

        # Create example (using dspy.Example for compatibility, but can use dict)
        example = dspy.Example(
            article_text=article_text,
            flood_mentioned=label
        ).with_inputs('article_text')

        examples.append(example)

    return examples


def load_unlabeled_data(filepath: str) -> list:
    """
    Load unlabeled data for pseudo-labeling.

    MODIFY THIS if your unlabeled data has a different format.
    """
    with open(filepath, 'r') as f:
        raw_data = json.load(f)

    return [
        {'article_text': item.get('full_text', item.get('article_text', ''))}
        for item in raw_data
    ]


# =============================================================================
# WEIGHTED TRAINER FOR HIGH RECALL
# =============================================================================

class HighRecallTrainer(Trainer):
    """Custom trainer with weighted loss for high recall."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            print(f"  Class weights: negative={class_weights[0]:.3f}, positive={class_weights[1]:.3f}")
            print(f"  Positive weight is {class_weights[1]/class_weights[0]:.2f}x higher")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


def calculate_class_weights(examples, multiplier=1.1):
    """
    Calculate class weights for imbalanced data with recall boost.

    Args:
        examples: Training examples with .flood_mentioned attribute
        multiplier: Extra weight for positive class (higher = more recall)

    Returns:
        torch.Tensor of shape [2] with weights for [negative, positive]
    """
    num_positive = sum(1 for ex in examples if ex.flood_mentioned)
    num_negative = len(examples) - num_positive
    total = len(examples)

    # Inverse frequency weighting
    weight_negative = total / (2 * num_negative) if num_negative > 0 else 1.0
    weight_positive = total / (2 * num_positive) if num_positive > 0 else 1.0

    # Apply recall multiplier to positive class
    weight_positive *= multiplier

    # Normalize
    total_weight = weight_negative + weight_positive
    weight_negative = weight_negative / total_weight * 2
    weight_positive = weight_positive / total_weight * 2

    return torch.tensor([weight_negative, weight_positive], dtype=torch.float32)


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(eval_pred):
    """Calculate classification metrics with recall focus."""
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def find_threshold_for_target_recall(probs, labels, target_recall, config):
    """
    Find classification threshold that achieves target recall.

    Searches from high to low threshold, finding the highest threshold
    (best precision) that still meets the recall target.
    """
    best_threshold = None
    best_precision = 0

    for threshold in np.arange(config['threshold_max'], config['threshold_min'], -config['threshold_step']):
        pred_labels = (probs > threshold).astype(int)

        tp = np.sum((labels == 1) & (pred_labels == 1))
        fn = np.sum((labels == 1) & (pred_labels == 0))
        fp = np.sum((labels == 0) & (pred_labels == 1))
        tn = np.sum((labels == 0) & (pred_labels == 0))

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        if recall >= target_recall and precision > best_precision:
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
            best_filter_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Fallback to minimum threshold if target not achievable
    if best_threshold is None:
        best_threshold = config['threshold_min']
        pred_labels = (probs > best_threshold).astype(int)
        tp = np.sum((labels == 1) & (pred_labels == 1))
        fn = np.sum((labels == 1) & (pred_labels == 0))
        fp = np.sum((labels == 0) & (pred_labels == 1))
        tn = np.sum((labels == 0) & (pred_labels == 0))

        best_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        best_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        best_filter_rate = tn / (tn + fp) if (tn + fp) > 0 else 0

    return best_threshold, best_recall, best_precision, best_filter_rate


# =============================================================================
# DATASET PREPARATION
# =============================================================================

def examples_to_dataset(examples):
    """Convert examples to HuggingFace Dataset format."""
    return Dataset.from_dict({
        'text': [ex.article_text for ex in examples],
        'label': [1 if ex.flood_mentioned else 0 for ex in examples]
    })


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize a dataset."""
    def tokenize_fn(batch):
        return tokenizer(
            batch['text'],
            padding='max_length',
            truncation=True,
            max_length=max_length
        )
    return dataset.map(tokenize_fn, batched=True)


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train_model(train_examples, val_examples, output_dir, config, iteration=0):
    """
    Train a single iteration of the model.

    Uses validation set for model selection (NOT test set).
    """
    print(f"\n{'='*70}")
    print(f"TRAINING ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Training examples: {len(train_examples)}")
    print(f"Validation examples: {len(val_examples)}")

    # Calculate class weights
    class_weights = calculate_class_weights(
        train_examples,
        multiplier=config['class_weight_multiplier']
    )

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['bert_model'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['bert_model'],
        num_labels=2,
        problem_type="single_label_classification"
    )

    # Prepare datasets
    train_dataset = tokenize_dataset(
        examples_to_dataset(train_examples),
        tokenizer,
        config['max_length']
    )
    val_dataset = tokenize_dataset(
        examples_to_dataset(val_examples),
        tokenizer,
        config['max_length']
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=config['learning_rate'],
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",  # Use F1 for checkpoint selection
        greater_is_better=True,
        save_total_limit=2,
        seed=RANDOM_SEED,
    )

    # Create trainer
    trainer = HighRecallTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # Validation, NOT test
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Evaluate on validation set
    val_results = trainer.evaluate(val_dataset)
    print(f"\nValidation Results:")
    print(f"  Accuracy:  {val_results['eval_accuracy']:.3f}")
    print(f"  Precision: {val_results['eval_precision']:.3f}")
    print(f"  Recall:    {val_results['eval_recall']:.3f}")
    print(f"  F1:        {val_results['eval_f1']:.3f}")

    # Find optimal threshold using validation set
    predictions = trainer.predict(val_dataset)
    probs = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    true_labels = predictions.label_ids

    threshold, recall, precision, filter_rate = find_threshold_for_target_recall(
        probs, true_labels, config['target_recall'], config
    )

    print(f"\nOptimal Threshold for {config['target_recall']:.0%}+ Recall:")
    print(f"  Threshold:   {threshold:.3f}")
    print(f"  Recall:      {recall:.1%} {'✓' if recall >= config['target_recall'] else '✗'}")
    print(f"  Precision:   {precision:.1%}")
    print(f"  Filter rate: {filter_rate:.1%}")

    return model, tokenizer, threshold, recall, precision, filter_rate


# =============================================================================
# PSEUDO-LABELING
# =============================================================================

def generate_pseudo_labels(model, tokenizer, unlabeled_pool, config):
    """
    Generate pseudo-labels from high-confidence predictions.

    Only adds examples where the model is very confident (>95% by default).
    Balances positive and negative pseudo-labels.
    """
    print(f"\n{'='*70}")
    print("GENERATING PSEUDO-LABELS")
    print(f"{'='*70}")
    print(f"Unlabeled pool: {len(unlabeled_pool):,}")

    confidence_threshold = config['pseudo_label_confidence']
    max_per_iteration = config['max_pseudo_labels_per_iteration']
    batch_size = config['batch_size']
    max_length = config['max_length']

    texts = [item['article_text'] for item in unlabeled_pool]

    model.eval()
    device = next(model.parameters()).device
    all_probs = []

    print("Predicting on unlabeled data...")
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)

        if (i + batch_size) % 1000 == 0:
            print(f"  Processed {min(i+batch_size, len(texts)):,}/{len(texts):,}")

    # Select high-confidence predictions
    high_conf_positive = [(idx, prob) for idx, prob in enumerate(all_probs) if prob > confidence_threshold]
    high_conf_negative = [(idx, prob) for idx, prob in enumerate(all_probs) if prob < (1 - confidence_threshold)]

    print(f"\nHigh-confidence predictions:")
    print(f"  Positive: {len(high_conf_positive)} (prob > {confidence_threshold})")
    print(f"  Negative: {len(high_conf_negative)} (prob < {1-confidence_threshold})")

    # Balance classes
    max_per_class = min(
        max_per_iteration // 2,
        len(high_conf_positive),
        len(high_conf_negative)
    )

    pseudo_labeled = []

    # Add positive examples
    selected_positive = sorted(high_conf_positive, key=lambda x: x[1], reverse=True)[:max_per_class]
    for idx, prob in selected_positive:
        article = unlabeled_pool[idx]
        pseudo_labeled.append(
            dspy.Example(
                article_text=article['article_text'],
                flood_mentioned=True
            ).with_inputs('article_text')
        )

    # Add negative examples
    selected_negative = sorted(high_conf_negative, key=lambda x: x[1])[:max_per_class]
    for idx, prob in selected_negative:
        article = unlabeled_pool[idx]
        pseudo_labeled.append(
            dspy.Example(
                article_text=article['article_text'],
                flood_mentioned=False
            ).with_inputs('article_text')
        )

    print(f"\nSelected: {len(selected_positive)} positive, {len(selected_negative)} negative")

    # Remove used articles from pool
    used_indices = set([idx for idx, _ in selected_positive] + [idx for idx, _ in selected_negative])
    remaining_pool = [item for idx, item in enumerate(unlabeled_pool) if idx not in used_indices]

    return pseudo_labeled, remaining_pool


# =============================================================================
# FINAL TEST EVALUATION
# =============================================================================

def evaluate_on_test_set(model, tokenizer, test_data, threshold, config):
    """
    Evaluate model on held-out test set for unbiased metrics.

    This should only be called ONCE at the very end of training.
    """
    print(f"\n{'='*70}")
    print("FINAL TEST SET EVALUATION (UNBIASED)")
    print(f"{'='*70}")

    model.eval()
    device = next(model.parameters()).device

    test_texts = [ex.article_text for ex in test_data]
    test_labels = np.array([1 if ex.flood_mentioned else 0 for ex in test_data])
    test_probs = []

    print(f"Evaluating on {len(test_data)} held-out test examples...")

    for i in range(0, len(test_texts), config['batch_size']):
        batch_texts = test_texts[i:i+config['batch_size']]
        with torch.no_grad():
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                truncation=True,
                max_length=config['max_length'],
                padding=True
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)

    test_probs = np.array(test_probs)
    test_preds = (test_probs > threshold).astype(int)

    # Calculate metrics
    tp = np.sum((test_labels == 1) & (test_preds == 1))
    fn = np.sum((test_labels == 1) & (test_preds == 0))
    tn = np.sum((test_labels == 0) & (test_preds == 0))
    fp = np.sum((test_labels == 0) & (test_preds == 1))

    test_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall) if (test_precision + test_recall) > 0 else 0

    print(f"\nUnbiased Test Metrics:")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Recall:    {test_recall:.1%} {'✓' if test_recall >= config['target_recall'] else '✗'}")
    print(f"  Precision: {test_precision:.1%}")
    print(f"  F1:        {test_f1:.3f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}, FN={fn}")
    print(f"    FP={fp}, TN={tn}")

    return {
        'recall': test_recall,
        'precision': test_precision,
        'f1': test_f1,
        'threshold': threshold,
        'confusion_matrix': {'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn)}
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    config = CONFIG

    # Create output directories
    output_dir = Path(config['output_dir'])
    results_dir = Path(config['results_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    with open(config['train_data_path'], 'r') as f:
        labeled_all = prepare_data(json.load(f))
    print(f"Labeled data: {len(labeled_all)} examples")

    with open(config['test_data_path'], 'r') as f:
        test_data = prepare_data(json.load(f))
    print(f"Test data (held out): {len(test_data)} examples")

    unlabeled_pool = load_unlabeled_data(config['unlabeled_data_path'])
    print(f"Unlabeled data: {len(unlabeled_pool):,} examples")

    # -------------------------------------------------------------------------
    # Split into Train/Validation
    # -------------------------------------------------------------------------
    labels_for_split = [1 if ex.flood_mentioned else 0 for ex in labeled_all]
    labeled_train, validation_data = train_test_split(
        labeled_all,
        test_size=config['validation_split'],
        random_state=RANDOM_SEED,
        stratify=labels_for_split
    )
    print(f"\nTrain/Val Split:")
    print(f"  Training:   {len(labeled_train)} examples")
    print(f"  Validation: {len(validation_data)} examples")

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)

    current_train_set = labeled_train.copy()
    remaining_unlabeled = unlabeled_pool.copy()
    iteration_results = []

    for iteration in range(config['num_iterations']):
        # Train model
        iter_output_dir = output_dir / f'iteration_{iteration}'
        model, tokenizer, threshold, recall, precision, filter_rate = train_model(
            current_train_set, validation_data, iter_output_dir, config, iteration
        )

        iteration_results.append({
            'iteration': iteration,
            'train_size': len(current_train_set),
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'filter_rate': filter_rate,
        })

        # Generate pseudo-labels (except on last iteration)
        if iteration < config['num_iterations'] - 1:
            pseudo_labeled, remaining_unlabeled = generate_pseudo_labels(
                model, tokenizer, remaining_unlabeled, config
            )

            if len(pseudo_labeled) == 0:
                print("\nNo pseudo-labels generated. Stopping early.")
                break

            current_train_set.extend(pseudo_labeled)
            print(f"\nUpdated training set: {len(current_train_set)} examples")

        # Save model
        model.save_pretrained(str(iter_output_dir))
        tokenizer.save_pretrained(str(iter_output_dir))

        with open(iter_output_dir / 'threshold_info.json', 'w') as f:
            json.dump({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'filter_rate': filter_rate,
            }, f, indent=2)

    # -------------------------------------------------------------------------
    # Final Test Evaluation
    # -------------------------------------------------------------------------
    # Find best iteration (highest recall meeting target, or highest overall)
    best_iter = max(
        range(len(iteration_results)),
        key=lambda i: (
            iteration_results[i]['recall'] >= config['target_recall'],
            iteration_results[i]['precision'] if iteration_results[i]['recall'] >= config['target_recall'] else iteration_results[i]['recall']
        )
    )

    best_model_dir = output_dir / f'iteration_{best_iter}'
    best_tokenizer = AutoTokenizer.from_pretrained(str(best_model_dir))
    best_model = AutoModelForSequenceClassification.from_pretrained(str(best_model_dir))
    best_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    test_results = evaluate_on_test_set(
        best_model,
        best_tokenizer,
        test_data,
        iteration_results[best_iter]['threshold'],
        config
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    df = pd.DataFrame(iteration_results)
    print("\nIteration Summary (validation metrics):")
    print(df.to_string(index=False))

    print(f"\nBest Model: iteration_{best_iter}")
    print(f"  Threshold: {iteration_results[best_iter]['threshold']:.3f}")
    print(f"\nUnbiased Test Performance:")
    print(f"  Recall:    {test_results['recall']:.1%}")
    print(f"  Precision: {test_results['precision']:.1%}")

    # Save final results
    with open(results_dir / 'final_results.json', 'w') as f:
        json.dump({
            'best_iteration': best_iter,
            'iteration_results': iteration_results,
            'test_results': test_results,
            'config': {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
        }, f, indent=2)

    print(f"\nResults saved to: {results_dir / 'final_results.json'}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
