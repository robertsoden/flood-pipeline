"""
Stage 3: Validate Extraction Model
Test the trained model on held-out test set
"""
import sys
from pathlib import Path
import dspy

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import MODEL_CONFIG, PROJECT_ROOT as PROJ_ROOT
from shared.utils import load_json, setup_logging
from stage3.extraction_signature import LocationDateExtractor


def normalize_location(loc: str) -> str:
    """Normalize location for comparison"""
    return loc.lower().strip().replace(',', '').replace(';', '')


def normalize_date(date_str: str) -> str:
    """Normalize date for comparison"""
    return date_str.strip().replace('/', '-')


def evaluate_extraction(pred, true_loc: str, true_date: str) -> dict:
    """
    Evaluate extraction quality
    
    Returns dict with scores for location and date
    """
    scores = {
        'location_exact': 0.0,
        'location_partial': 0.0,
        'date_exact': 0.0,
        'date_partial': 0.0
    }
    
    # Location evaluation
    pred_loc = normalize_location(pred.primary_location)
    norm_true_loc = normalize_location(true_loc)
    
    if pred_loc == norm_true_loc:
        scores['location_exact'] = 1.0
        scores['location_partial'] = 1.0
    elif pred_loc in norm_true_loc or norm_true_loc in pred_loc:
        scores['location_partial'] = 0.7
    else:
        # Check for partial overlap in parts
        pred_parts = set(pred_loc.split())
        true_parts = set(norm_true_loc.split())
        overlap = pred_parts & true_parts
        if overlap:
            scores['location_partial'] = 0.3 * len(overlap) / len(true_parts)
    
    # Date evaluation (if available)
    if true_date and true_date != 'unknown':
        pred_date = normalize_date(pred.flood_date)
        norm_true_date = normalize_date(true_date)
        
        if pred_date == norm_true_date:
            scores['date_exact'] = 1.0
            scores['date_partial'] = 1.0
        elif len(pred_date) >= 7 and len(norm_true_date) >= 7:
            # Month match (YYYY-MM)
            if pred_date[:7] == norm_true_date[:7]:
                scores['date_partial'] = 0.7
        elif len(pred_date) >= 4 and len(norm_true_date) >= 4:
            # Year match
            if pred_date[:4] == norm_true_date[:4]:
                scores['date_partial'] = 0.4
    
    return scores


def validate_model(
    model_file: Path,
    test_file: Path = None,
    max_examples: int = None
):
    """
    Validate trained model on test set
    
    Args:
        model_file: Path to trained model
        test_file: Path to test examples (default: use dev set)
        max_examples: Maximum number of examples to test (default: all)
    """
    logger = setup_logging('stage3_validation')
    
    logger.info("=" * 70)
    logger.info("STAGE 3: MODEL VALIDATION")
    logger.info("=" * 70)
    
    # Configure LLM
    logger.info(f"\n[1/4] Configuring LLM: {MODEL_CONFIG['name']}")
    lm = dspy.LM(
        model=MODEL_CONFIG['name'],
        api_base=MODEL_CONFIG['api_base'],
        api_key=MODEL_CONFIG['api_key']
    )
    dspy.configure(lm=lm)
    
    # Load model
    logger.info(f"\n[2/4] Loading model from {model_file}")
    if model_file.exists():
        extractor = LocationDateExtractor()
        extractor.load(str(model_file))
        logger.info("Model loaded successfully")
    else:
        logger.error(f"Model file not found: {model_file}")
        logger.info("Using baseline (untrained) model")
        extractor = LocationDateExtractor()
    
    # Load test data
    if test_file is None:
        test_file = PROJ_ROOT / 'stage3' / 'data' / 'dev_examples.json'
    
    logger.info(f"\n[3/4] Loading test data from {test_file}")
    if not test_file.exists():
        logger.error(f"Test file not found: {test_file}")
        logger.info("Please run prepare_training_data.py first")
        return
    
    test_data = load_json(test_file)
    
    if max_examples:
        test_data = test_data[:max_examples]
    
    logger.info(f"Testing on {len(test_data)} examples")
    
    # Run validation
    logger.info(f"\n[4/4] Running validation...")
    
    results = []
    location_exact_scores = []
    location_partial_scores = []
    date_exact_scores = []
    date_partial_scores = []
    
    for i, example in enumerate(test_data, 1):
        try:
            # Run extraction
            pred = extractor(
                article_text=example['article_text'],
                publication_date=example['publication_date']
            )
            
            # Evaluate
            scores = evaluate_extraction(
                pred,
                example['primary_location'],
                example.get('flood_date', 'unknown')
            )
            
            results.append({
                'example': i,
                'true_location': example['primary_location'],
                'pred_location': pred.primary_location,
                'true_date': example.get('flood_date', 'unknown'),
                'pred_date': pred.flood_date,
                'scores': scores
            })
            
            location_exact_scores.append(scores['location_exact'])
            location_partial_scores.append(scores['location_partial'])
            date_exact_scores.append(scores['date_exact'])
            date_partial_scores.append(scores['date_partial'])
            
            if i % 10 == 0 or i == len(test_data):
                logger.info(f"Progress: {i}/{len(test_data)}")
        
        except Exception as e:
            logger.error(f"Error on example {i}: {e}")
            results.append({
                'example': i,
                'error': str(e)
            })
    
    # Calculate metrics
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 70)
    
    loc_exact_avg = sum(location_exact_scores) / len(location_exact_scores) * 100
    loc_partial_avg = sum(location_partial_scores) / len(location_partial_scores) * 100
    date_exact_avg = sum(date_exact_scores) / len(date_exact_scores) * 100 if date_exact_scores else 0
    date_partial_avg = sum(date_partial_scores) / len(date_partial_scores) * 100 if date_partial_scores else 0
    
    logger.info(f"\nLocation Extraction:")
    logger.info(f"  Exact match:   {loc_exact_avg:.1f}%")
    logger.info(f"  Partial match: {loc_partial_avg:.1f}%")
    
    logger.info(f"\nDate Extraction:")
    logger.info(f"  Exact match:   {date_exact_avg:.1f}%")
    logger.info(f"  Partial match: {date_partial_avg:.1f}%")
    
    # Show examples
    logger.info(f"\n" + "=" * 70)
    logger.info("SAMPLE RESULTS")
    logger.info("=" * 70)
    
    # Perfect matches
    perfect = [r for r in results if r.get('scores', {}).get('location_exact', 0) == 1.0]
    if perfect:
        logger.info(f"\n✓ PERFECT LOCATION MATCHES ({len(perfect)} total):")
        for r in perfect[:3]:
            logger.info(f"\n  Example {r['example']}:")
            logger.info(f"    True: {r['true_location']}")
            logger.info(f"    Pred: {r['pred_location']}")
            if r['true_date'] != 'unknown':
                logger.info(f"    Date: {r['true_date']} → {r['pred_date']}")
    
    # Partial matches
    partial = [r for r in results 
               if 0 < r.get('scores', {}).get('location_partial', 0) < 1.0]
    if partial:
        logger.info(f"\n⚠ PARTIAL LOCATION MATCHES ({len(partial)} total):")
        for r in partial[:3]:
            logger.info(f"\n  Example {r['example']}:")
            logger.info(f"    True: {r['true_location']}")
            logger.info(f"    Pred: {r['pred_location']}")
            logger.info(f"    Score: {r['scores']['location_partial']:.2f}")
    
    # Misses
    misses = [r for r in results if r.get('scores', {}).get('location_partial', 0) == 0]
    if misses:
        logger.info(f"\n✗ LOCATION MISSES ({len(misses)} total):")
        for r in misses[:3]:
            logger.info(f"\n  Example {r['example']}:")
            logger.info(f"    True: {r['true_location']}")
            logger.info(f"    Pred: {r['pred_location']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 70)
    
    # Overall assessment
    if loc_partial_avg >= 90:
        logger.info("\n✓ EXCELLENT: Model achieves >90% location accuracy!")
    elif loc_partial_avg >= 80:
        logger.info("\n✓ GOOD: Model achieves >80% location accuracy")
    elif loc_partial_avg >= 70:
        logger.info("\n⚠ ACCEPTABLE: Model achieves >70% location accuracy")
    else:
        logger.info("\n✗ NEEDS IMPROVEMENT: Location accuracy <70%")
        logger.info("  Consider: more training data, adjusting DSPy config")
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate Stage 3 extraction model')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--test-file', type=str, help='Path to test examples')
    parser.add_argument('--max-examples', type=int, help='Max examples to test')
    
    args = parser.parse_args()
    
    model_file = Path(args.model) if args.model else PROJ_ROOT / 'stage3' / 'models' / 'optimized_extractor.json'
    test_file = Path(args.test_file) if args.test_file else None
    
    validate_model(model_file, test_file, args.max_examples)


if __name__ == '__main__':
    main()
