"""
Stage 3: Train and Optimize Extraction Model
Uses DSPy to optimize the location and date extraction prompts
"""
import sys
from pathlib import Path
import dspy
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import MODEL_CONFIG, STAGE3_CONFIG
from shared.utils import load_json, save_json, setup_logging
from stage3.signature import LocationDateExtractor, LocationDateExtraction


def example_to_dspy(example: dict) -> dspy.Example:
    """Convert training example dict to DSPy Example"""
    return dspy.Example(
        article_text=example['article_text'],
        publication_date=example['publication_date'],
        primary_location=example['primary_location'],
        location_confidence=example['location_confidence'],
        location_reasoning=example['location_reasoning'],
        flood_date=example['flood_date'],
        date_precision=example['date_precision'],
        date_confidence=example['date_confidence'],
        date_reasoning=example['date_reasoning']
    ).with_inputs('article_text', 'publication_date')


def metric(example: dspy.Example, prediction, trace=None) -> float:
    """
    Evaluation metric for extraction quality
    
    Checks if predicted location and date match the ground truth
    """
    score = 0.0
    
    # Location match (most important)
    pred_loc = prediction.primary_location.lower().strip()
    true_loc = example.primary_location.lower().strip()
    
    # Exact match
    if pred_loc == true_loc:
        score += 0.5
    # Partial match (one contains the other)
    elif pred_loc in true_loc or true_loc in pred_loc:
        score += 0.3
    # Check if key parts match (split on commas/semicolons)
    else:
        pred_parts = set(p.strip() for p in pred_loc.replace(';', ',').split(','))
        true_parts = set(t.strip() for t in true_loc.replace(';', ',').split(','))
        overlap = pred_parts & true_parts
        if overlap:
            score += 0.2
    
    # Date match (less important since some are imprecise)
    if example.flood_date != 'unknown':
        pred_date = prediction.flood_date.strip()
        true_date = example.flood_date.strip()
        
        # Exact match
        if pred_date == true_date:
            score += 0.5
        # Year-month match (if dates are at month or year precision)
        elif len(pred_date) >= 7 and len(true_date) >= 7:
            if pred_date[:7] == true_date[:7]:  # YYYY-MM
                score += 0.3
        # Year match only
        elif len(pred_date) >= 4 and len(true_date) >= 4:
            if pred_date[:4] == true_date[:4]:  # YYYY
                score += 0.2
    
    return score


def train_and_optimize(
    train_file: Path,
    dev_file: Path,
    output_dir: Path,
    num_trials: int = 10
):
    """
    Train and optimize the extraction model using DSPy
    
    Args:
        train_file: Path to training examples JSON
        dev_file: Path to development examples JSON
        output_dir: Directory to save optimized model
        num_trials: Number of optimization trials
    """
    logger = setup_logging('stage3_optimization')
    
    logger.info("=" * 70)
    logger.info("STAGE 3: MODEL OPTIMIZATION")
    logger.info("=" * 70)
    
    # Load training data
    logger.info(f"\n[1/6] Loading training data...")
    train_data = load_json(train_file)
    dev_data = load_json(dev_file)
    
    logger.info(f"Training examples: {len(train_data)}")
    logger.info(f"Development examples: {len(dev_data)}")
    
    # Convert to DSPy examples
    train_examples = [example_to_dspy(ex) for ex in train_data]
    dev_examples = [example_to_dspy(ex) for ex in dev_data]
    
    # Configure LLM
    logger.info(f"\n[2/6] Configuring LLM: {MODEL_CONFIG['name']}")
    logger.info(f"Temperature: {STAGE3_CONFIG['temperature_optimization']} (optimization mode)")
    lm = dspy.LM(
        model=MODEL_CONFIG['name'],
        api_base=MODEL_CONFIG['api_base'],
        api_key=MODEL_CONFIG['api_key'],
        temperature=STAGE3_CONFIG['temperature_optimization']  # Higher for optimization
    )
    dspy.configure(lm=lm)
    
    # Create baseline model
    logger.info(f"\n[3/6] Creating baseline model...")
    baseline = LocationDateExtractor()
    
    # Evaluate baseline on dev set (sample)
    logger.info(f"\n[4/6] Evaluating baseline on sample...")
    sample_size = min(30, len(dev_examples))  # Use larger sample for reliable evaluation
    baseline_scores = []
    for i, example in enumerate(dev_examples[:sample_size]):
        try:
            prediction = baseline(
                article_text=example.article_text,
                publication_date=example.publication_date
            )
            score = metric(example, prediction)
            baseline_scores.append(score)
            logger.info(f"  Example {i+1}/{sample_size}: score = {score:.2f}")
        except Exception as e:
            logger.warning(f"  Example {i+1} failed: {e}")
            baseline_scores.append(0.0)
    
    avg_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
    logger.info(f"Baseline average score: {avg_baseline:.3f}")
    
    # Optimize using MIPROv2
    logger.info(f"\n[5/6] Optimizing with MIPROv2...")
    logger.info(f"Auto mode: light (~10 trials for faster optimization)")
    logger.info(f"Max bootstrapped demos: {STAGE3_CONFIG.get('max_bootstrapped_demos', 3)}")
    logger.info(f"Max labeled demos: {STAGE3_CONFIG.get('max_labeled_demos', 3)}")
    logger.info(f"Num threads: {STAGE3_CONFIG.get('num_threads', 8)}")

    try:
        from dspy.teleprompt import MIPROv2

        optimizer = MIPROv2(
            metric=metric,
            auto="light",  # ~10 trials, faster optimization
            num_threads=STAGE3_CONFIG.get('num_threads', 8),
            verbose=True
        )

        # Use more training examples for better optimization
        train_subset_size = min(50, len(train_examples))  # Use up to 50 examples
        logger.info(f"Using {train_subset_size} training examples")

        # Optimize on training set
        optimized = optimizer.compile(
            baseline,
            trainset=train_examples[:train_subset_size],
            max_bootstrapped_demos=STAGE3_CONFIG.get('max_bootstrapped_demos', 3),
            max_labeled_demos=STAGE3_CONFIG.get('max_labeled_demos', 3)
        )
        
        logger.info("Optimization complete!")
        
        # Evaluate optimized model
        logger.info(f"\n[6/6] Evaluating optimized model on dev set...")
        optimized_scores = []
        for i, example in enumerate(dev_examples[:sample_size]):
            try:
                prediction = optimized(
                    article_text=example.article_text,
                    publication_date=example.publication_date
                )
                score = metric(example, prediction)
                optimized_scores.append(score)
                logger.info(f"  Example {i+1}/{sample_size}: score = {score:.2f}")
            except Exception as e:
                logger.warning(f"  Example {i+1} failed: {e}")
                optimized_scores.append(0.0)
        
        avg_optimized = sum(optimized_scores) / len(optimized_scores) if optimized_scores else 0.0
        logger.info(f"Optimized average score: {avg_optimized:.3f}")
        logger.info(f"Improvement: {avg_optimized - avg_baseline:+.3f}")
        
        # Save optimized model
        output_dir.mkdir(exist_ok=True, parents=True)
        optimized_path = output_dir / 'optimized_extractor.json'
        
        logger.info(f"\nSaving optimized model to {optimized_path}")
        optimized.save(str(optimized_path))
        
        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"\nBaseline score:  {avg_baseline:.3f}")
        logger.info(f"Optimized score: {avg_optimized:.3f}")
        logger.info(f"Improvement:     {avg_optimized - avg_baseline:+.3f} ({(avg_optimized/avg_baseline - 1)*100:+.1f}%)")
        
        return optimized
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        logger.info("Using baseline model instead")
        
        # Save baseline as fallback
        baseline_path = output_dir / 'baseline_extractor.json'
        logger.info(f"Saving baseline model to {baseline_path}")
        baseline.save(str(baseline_path))
        
        return baseline


def main():
    """Main function"""
    train_file = PROJECT_ROOT / 'stage3' / 'data' / 'train_examples.json'
    dev_file = PROJECT_ROOT / 'stage3' / 'data' / 'dev_examples.json'
    output_dir = PROJECT_ROOT / 'stage3' / 'models'
    
    # Check if training data exists
    if not train_file.exists():
        print(f"Training data not found at {train_file}")
        print("Please run prepare_training_data.py first")
        return
    
    # Run optimization
    train_and_optimize(train_file, dev_file, output_dir)


if __name__ == '__main__':
    main()
