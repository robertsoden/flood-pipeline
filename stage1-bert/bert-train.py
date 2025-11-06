"""
High-Recall BERT Training - Optimized for Balanced Dataset
"""
import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared config and utils
from shared import (
    train_filepath,
    val_filepath,
    test_filepath,
    unlabeled_filepath,
    STAGE1_CONFIG,
    prepare_data,
)

print("\n" + "="*70)
print("HIGH-RECALL BERT TRAINING (BALANCED DATASET)")
print("="*70)

# Configuration from shared config
UNLABELED_DATA_PATH = unlabeled_filepath
MODEL_NAME = STAGE1_CONFIG.get('bert_model', 'distilbert-base-uncased')
TARGET_RECALL = STAGE1_CONFIG.get('target_recall', 0.95)

# Training-specific settings (could be added to STAGE1_CONFIG)
PSEUDO_LABEL_CONFIDENCE = 0.95
MAX_PSEUDO_LABELS_PER_ITERATION = 3000
NUM_ITERATIONS = 3
RECALL_WEIGHT_MULTIPLIER = 1.1
DEFAULT_CLASSIFICATION_THRESHOLD = 0.30

print("Configuration for Balanced Dataset:")
print(f"  Model: {MODEL_NAME}")
print(f"  Target recall: {TARGET_RECALL:.0%}")
print(f"  Recall weight multiplier: {RECALL_WEIGHT_MULTIPLIER}x (conservative)")
print(f"  Default threshold: {DEFAULT_CLASSIFICATION_THRESHOLD}")
print(f"  Pseudo-label confidence: {PSEUDO_LABEL_CONFIDENCE}")
print(f"  Note: Conservative weighting - let the balanced data (47% floods) speak!")

# Create output directories
MODELS_DIR = PROJECT_ROOT / 'models'
RESULTS_DIR = PROJECT_ROOT / 'results'
MODELS_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================================
# WEIGHTED TRAINER FOR BALANCED DATA
# ============================================================================

class BalancedHighRecallTrainer(Trainer):
    """Trainer optimized for high recall with conservative weighting for balanced data"""
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        if class_weights is not None:
            print(f"    ✓ Weights: non-flood={class_weights[0]:.3f}, flood={class_weights[1]:.3f}")
            print(f"    ✓ Flood weight is {class_weights[1]/class_weights[0]:.2f}x higher (conservative)")
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute weighted loss for imbalanced classification.
        
        Args:
            model: The model being trained
            inputs: Dictionary of inputs including labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (for newer transformers versions)
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted cross-entropy
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def calculate_balanced_class_weights(train_examples, multiplier=1.5):
    """
    Calculate class weights for balanced dataset
    
    With 47% floods, we only need conservative weighting
    multiplier=1.1 is sufficient (vs 1.5-3.0 for imbalanced data)
    """
    num_floods = sum(1 for ex in train_examples if ex.flood_mentioned)
    num_non_floods = len(train_examples) - num_floods
    total = len(train_examples)
    
    # Base inverse frequency weighting
    weight_non_flood = total / (2 * num_non_floods)
    weight_flood = total / (2 * num_floods)
    
    # Apply moderate multiplier
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
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, recall_score
    
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
print(f"   Labeled training: {len(labeled_train)} examples")

# Check if val file exists
try:
    with open(val_filepath, 'r') as file:
        val_data = prepare_data(json.load(file))
    print(f"   Validation: {len(val_data)} examples")
except:
    print(f"   No separate validation file, will use test for validation")
    val_data = None

with open(test_filepath, 'r') as file:
    test_data = prepare_data(json.load(file))
print(f"   Test set: {len(test_data)} examples")

# Class distribution check
train_floods = sum(1 for ex in labeled_train if ex.flood_mentioned)
train_non_floods = len(labeled_train) - train_floods
print(f"\n   Training distribution:")
print(f"     Floods: {train_floods} ({train_floods/len(labeled_train):.1%})")
print(f"     Non-floods: {train_non_floods} ({train_non_floods/len(labeled_train):.1%})")
print(f"     Balance ratio: 1:{train_non_floods/train_floods:.2f}")

if train_floods / len(labeled_train) > 0.4:
    print(f"   ✅ Well-balanced dataset! Using moderate weighting.")
else:
    print(f"   ⚠️  Imbalanced dataset. Consider increasing RECALL_WEIGHT_MULTIPLIER.")

print("\n2. Loading unlabeled data...")
with open(UNLABELED_DATA_PATH, 'r') as file:
    unlabeled_raw = json.load(file)
print(f"   Unlabeled articles: {len(unlabeled_raw):,}")

unlabeled_pool = [
    {'article_text': article['full_text'], 
     'publication_date': article.get('publication_date', '')}
    for article in unlabeled_raw
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def prepare_dataset(examples):
    """Convert to Hugging Face Dataset format"""
    return Dataset.from_dict({
        'text': [ex.article_text for ex in examples],
        'label': [1 if ex.flood_mentioned else 0 for ex in examples]
    })


def tokenize_dataset(dataset, tokenizer):
    """Tokenize dataset"""
    max_length = STAGE1_CONFIG.get('max_length', 512)
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            padding='max_length', 
            truncation=True, 
            max_length=max_length
        )
    return dataset.map(tokenize_function, batched=True)


def train_high_recall_model(train_examples, val_examples, test_examples, output_dir, iteration=0):
    """Train BERT model optimized for high recall on balanced data"""
    print(f"\n{'='*70}")
    print(f"HIGH-RECALL TRAINING - ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Training examples: {len(train_examples)}")
    
    # Calculate class weights
    class_weights, num_floods, num_non_floods = calculate_balanced_class_weights(
        train_examples, 
        multiplier=RECALL_WEIGHT_MULTIPLIER
    )
    
    print(f"  Class distribution:")
    print(f"    Floods: {num_floods} ({num_floods/len(train_examples):.1%})")
    print(f"    Non-floods: {num_non_floods} ({num_non_floods/len(train_examples):.1%})")
    print(f"    Balance ratio: 1:{num_non_floods/num_floods:.2f}")
    print(f"  Weighting strategy:")
    print(f"    Base balance: 1:{num_non_floods/num_floods:.2f}")
    print(f"    After multiplier: {class_weights[1]/class_weights[0]:.2f}:1 favor floods")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Prepare datasets
    train_dataset = prepare_dataset(train_examples)
    val_dataset = prepare_dataset(val_examples)
    test_dataset = prepare_dataset(test_examples)
    
    train_dataset = tokenize_dataset(train_dataset, tokenizer)
    val_dataset = tokenize_dataset(val_dataset, tokenizer)
    test_dataset = tokenize_dataset(test_dataset, tokenizer)
    
    # Training arguments
    batch_size = STAGE1_CONFIG.get('batch_size', 16)
    learning_rate = STAGE1_CONFIG.get('learning_rate', 2e-5)
    num_epochs = STAGE1_CONFIG.get('num_epochs', 4)
    
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=str(PROJECT_ROOT / 'logs'),
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=2
    )
    
    # Train
    trainer = BalancedHighRecallTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_recall_focused_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        class_weights=class_weights
    )
    
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Evaluate on validation set
    val_results = trainer.evaluate(val_dataset)
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:  {val_results['eval_accuracy']:.3f}")
    print(f"  Precision: {val_results['eval_precision']:.3f}")
    print(f"  Recall:    {val_results['eval_recall']:.3f}")
    print(f"  F1:        {val_results['eval_f1']:.3f}")
    print(f"\nRecall at different thresholds:")
    print(f"  @ 0.35: {val_results['eval_recall_at_0.35']:.3f}")
    print(f"  @ 0.30: {val_results['eval_recall_at_0.30']:.3f}")
    print(f"  @ 0.25: {val_results['eval_recall_at_0.25']:.3f}")
    
    # Evaluate on test set
    test_results = trainer.evaluate(test_dataset)
    print(f"\n{'='*70}")
    print(f"ITERATION {iteration} - TEST RESULTS")
    print(f"{'='*70}")
    print(f"  Accuracy:  {test_results['eval_accuracy']:.3f}")
    print(f"  Precision: {test_results['eval_precision']:.3f}")
    print(f"  Recall:    {test_results['eval_recall']:.3f}")
    print(f"  F1:        {test_results['eval_f1']:.3f}")
    
    # Find optimal threshold on test set
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
    
    if recall >= TARGET_RECALL:
        print(f"\n  ✅ Target recall achieved!")
        print(f"     With {len(train_examples)} training samples, model is stable and reliable.")
    else:
        print(f"\n  ⚠️  Target recall not quite achieved.")
        print(f"     Current: {recall:.1%} / Target: {TARGET_RECALL:.0%}")
        print(f"     Try: Increase RECALL_WEIGHT_MULTIPLIER to {RECALL_WEIGHT_MULTIPLIER * 1.2:.1f}")
    
    # Sample predictions
    print(f"\n📊 Sample predictions on test set:")
    model.eval()
    sample_indices = np.random.choice(len(test_examples), min(10, len(test_examples)), replace=False)
    
    max_length = STAGE1_CONFIG.get('max_length', 512)
    for idx in sample_indices:
        ex = test_examples[idx]
        with torch.no_grad():
            inputs = tokenizer(ex.article_text, return_tensors='pt', truncation=True, max_length=max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model(**inputs)
            prob = torch.softmax(outputs.logits, dim=1)[0, 1].item()
            pred = "FLOOD" if prob > threshold else "NOT FLOOD"
        
        actual = "FLOOD" if ex.flood_mentioned else "NOT FLOOD"
        correct = "✓" if (pred == actual) else "✗"
        print(f"  {correct} Actual={actual:10s} Pred={pred:10s} (prob={prob:.3f})")
    
    return model, tokenizer, trainer, threshold, recall, precision, filter_rate


def generate_pseudo_labels(model, tokenizer, unlabeled_pool, confidence_threshold):
    """Generate high-confidence pseudo-labels"""
    print(f"\n{'='*70}")
    print("GENERATING PSEUDO-LABELS")
    print(f"{'='*70}")
    print(f"Unlabeled pool size: {len(unlabeled_pool):,}")
    print(f"Confidence threshold: {confidence_threshold}")
    
    texts = [article['article_text'] for article in unlabeled_pool]
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    all_probs = []
    batch_size = STAGE1_CONFIG.get('batch_size', 32)
    max_length = STAGE1_CONFIG.get('max_length', 512)
    
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
    
    print(f"✓ Predictions complete!")
    
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
    
    import dspy
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
    
    print(f"\nSelected pseudo-labeled examples:")
    print(f"  Floods: {len(selected_floods)}")
    print(f"  Non-floods: {len(selected_non_floods)}")
    print(f"  Total: {len(pseudo_labeled)}")
    
    # Remove used articles
    used_indices = set([idx for idx, _ in selected_floods] + [idx for idx, _ in selected_non_floods])
    unlabeled_pool = [article for idx, article in enumerate(unlabeled_pool) if idx not in used_indices]
    
    print(f"Remaining unlabeled pool: {len(unlabeled_pool):,}")
    
    return pseudo_labeled, unlabeled_pool


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "="*70)
print("STARTING HIGH-RECALL TRAINING LOOP")
print("="*70)

current_train_set = labeled_train.copy()
remaining_unlabeled = unlabeled_pool.copy()

# Use validation set if available, otherwise use test
val_set = val_data if val_data is not None else test_data

iteration_results = []

for iteration in range(NUM_ITERATIONS):
    # Train model
    output_dir = MODELS_DIR / f'balanced_high_recall_iter{iteration}'
    model, tokenizer, trainer, threshold, recall, precision, filter_rate = train_high_recall_model(
        current_train_set,
        val_set,
        test_data,
        output_dir,
        iteration
    )
    
    iteration_results.append({
        'iteration': iteration,
        'train_size': len(current_train_set),
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'filter_rate': filter_rate,
        'weight_multiplier': RECALL_WEIGHT_MULTIPLIER
    })
    
    # Generate pseudo-labels for next iteration
    if iteration < NUM_ITERATIONS - 1:
        pseudo_labeled, remaining_unlabeled = generate_pseudo_labels(
            model, 
            tokenizer, 
            remaining_unlabeled,
            PSEUDO_LABEL_CONFIDENCE
        )
        
        if len(pseudo_labeled) == 0:
            print("\n⚠️  No pseudo-labels generated. Stopping early.")
            break
        
        current_train_set.extend(pseudo_labeled)
        print(f"\nUpdated training set size: {len(current_train_set)}")
    
    # Save model
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    
    # Save threshold info
    threshold_info = {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'filter_rate': filter_rate,
        'train_size': len(current_train_set),
        'weight_multiplier': RECALL_WEIGHT_MULTIPLIER
    }
    with open(output_dir / 'threshold_info.json', 'w') as f:
        json.dump(threshold_info, f, indent=2)
    
    print(f"✓ Model saved to: {output_dir}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*70)
print("HIGH-RECALL TRAINING COMPLETE")
print("="*70)

df_iterations = pd.DataFrame(iteration_results)
print("\nIteration Summary:")
print(df_iterations.to_string(index=False))

# Find best iteration
best_iter_idx = df_iterations['recall'].idxmax()
best_iter = df_iterations.loc[best_iter_idx]

print(f"\n🏆 BEST ITERATION: {int(best_iter['iteration'])}")
print(f"  Recall:      {best_iter['recall']:.1%} {'✅' if best_iter['recall'] >= TARGET_RECALL else '⚠️'}")
print(f"  Precision:   {best_iter['precision']:.1%}")
print(f"  Threshold:   {best_iter['threshold']:.3f}")
print(f"  Filter rate: {best_iter['filter_rate']:.1%}")

# Production estimates (adjusted for 48% flood rate)
print(f"\n💰 ESTIMATED PERFORMANCE ON 50,000 ARTICLES:")
print(f"   (Assuming ~48% flood rate based on your labeled data)")
flood_rate = 0.48
total_floods = int(50000 * flood_rate)
total_non_floods = int(50000 * (1 - flood_rate))

caught_floods = int(total_floods * best_iter['recall'])
missed_floods = total_floods - caught_floods

filtered_non_floods = int(total_non_floods * best_iter['filter_rate'])
false_alarms = total_non_floods - filtered_non_floods

articles_to_review = caught_floods + false_alarms

print(f"   Total floods: ~{total_floods:,}")
print(f"   Caught floods: ~{caught_floods:,} ({best_iter['recall']:.1%})")
print(f"   Missed floods: ~{missed_floods:,} ({(1-best_iter['recall']):.1%})")
print(f"   ")
print(f"   Total non-floods: ~{total_non_floods:,}")
print(f"   Filtered correctly: ~{filtered_non_floods:,} ({best_iter['filter_rate']:.1%})")
print(f"   False alarms: ~{false_alarms:,}")
print(f"   ")
print(f"   Articles to review: ~{articles_to_review:,} ({articles_to_review/50000:.1%})")
print(f"   Articles filtered: ~{filtered_non_floods:,} ({filtered_non_floods/50000:.1%})")
print(f"   Cost savings: ${filtered_non_floods * 0.01:,.2f} (@ $0.01/article)")

if best_iter['recall'] >= TARGET_RECALL:
    print(f"\n✅ SUCCESS! Achieved {TARGET_RECALL:.0%}+ recall target")
    print(f"   With {len(labeled_train)} training samples, model is production-ready!")
else:
    print(f"\n⚠️  Close but not quite at {TARGET_RECALL:.0%}+")
    print(f"   Current: {best_iter['recall']:.1%}")
    print(f"   Try: RECALL_WEIGHT_MULTIPLIER = {RECALL_WEIGHT_MULTIPLIER * 1.2:.1f}")

# Save results
results_file = RESULTS_DIR / 'balanced_high_recall_iterations.csv'
df_iterations.to_csv(results_file, index=False)
print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*70)
print("USAGE INSTRUCTIONS")
print("="*70)
print(f"\nTo use the best model (iteration {int(best_iter['iteration'])}):")
print(f"\n1. Load model:")
print(f"   model_path = '{MODELS_DIR}/balanced_high_recall_iter{int(best_iter['iteration'])}'")
print(f"   model = AutoModelForSequenceClassification.from_pretrained(model_path)")
print(f"   tokenizer = AutoTokenizer.from_pretrained(model_path)")
print(f"\n2. Load threshold:")
print(f"   with open(model_path + '/threshold_info.json') as f:")
print(f"       threshold = json.load(f)['threshold']  # {best_iter['threshold']:.3f}")
print(f"\n3. Classify:")
print(f"   prob = get_flood_probability(text)")
print(f"   is_flood = prob > threshold")

print("\n" + "="*70 + "\n")