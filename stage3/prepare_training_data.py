"""
Stage 3: Prepare Training Data
Convert annotated articles into DSPy training examples
"""
import sys
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import load_json, save_json, setup_logging


def normalize_date(date_str: str, pub_date: str) -> tuple[str, str]:
    """
    Normalize date annotation to YYYY-MM or YYYY format (NO DAYS)
    
    Returns:
        (normalized_date, precision)
        precision is either 'month' or 'year' (never 'day')
    """
    if not date_str or date_str.strip() == '':
        return ('unknown', 'unknown')
    
    date_str = date_str.strip()
    
    # Try to parse various formats
    # "February 1981" -> "1981-02"
    # "May 1996" -> "1996-05"
    # "1996" -> "1996"
    
    try:
        # If it has day-level precision, strip it to month only
        if '-' in date_str:
            parts = date_str.split('-')
            if len(parts) >= 2:
                # YYYY-MM or YYYY-MM-DD -> return YYYY-MM only
                return (f"{parts[0]}-{parts[1]}", 'month')
            else:
                return (parts[0], 'year')
        
        # Try "Month YYYY" format
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12'
        }
        
        parts = date_str.lower().split()
        if len(parts) == 2:
            month_name = parts[0]
            year = parts[1]
            if month_name in months:
                return (f"{year}-{months[month_name]}", 'month')
        
        # Just a year
        if date_str.isdigit() and len(date_str) == 4:
            return (date_str, 'year')
        
        # Can't parse
        return (date_str, 'unknown')
    
    except:
        return (date_str, 'unknown')


def prepare_training_data(
    input_file: Path,
    output_train: Path,
    output_dev: Path,
    train_split: float = 0.75
):
    """
    Prepare training data from annotated articles
    
    Args:
        input_file: Path to all_labeled_data.json
        output_train: Path to save training examples
        output_dev: Path to save development/validation examples
        train_split: Fraction of data to use for training (rest for dev)
    """
    logger = setup_logging('stage3_data_prep')
    
    logger.info("=" * 70)
    logger.info("STAGE 3: TRAINING DATA PREPARATION")
    logger.info("=" * 70)
    
    # Load data
    logger.info(f"Loading annotated data from {input_file}")
    data = load_json(input_file)
    logger.info(f"Loaded {len(data)} total articles")
    
    # Filter to flood articles with annotations
    flood_articles = [a for a in data if a.get('flood_mentioned')]
    logger.info(f"Found {len(flood_articles)} flood articles")
    
    # Separate by annotation type
    with_location = [a for a in flood_articles if a.get('location')]
    with_date = [a for a in flood_articles if a.get('flood_date')]
    with_both = [a for a in flood_articles if a.get('location') and a.get('flood_date')]
    
    logger.info(f"Articles with location annotations: {len(with_location)}")
    logger.info(f"Articles with date annotations: {len(with_date)}")
    logger.info(f"Articles with BOTH annotations: {len(with_both)}")
    
    # Create training examples
    # We'll use articles with BOTH location and date for best training
    # But we can also use single-annotation articles for specific tasks
    
    training_examples = []
    
    for article in with_both:
        # Normalize the annotations
        location = article.get('location', '').strip()
        flood_date = article.get('flood_date', '').strip()
        pub_date = article.get('publication_date', 'unknown')
        text = article.get('article_text', '')
        
        if not location and not flood_date:
            continue
        
        # For location: take the first one if multiple are listed (just use primary)
        primary_location = location.split(';')[0].strip() if ';' in location else location
        
        # Normalize date (month or year only, never day)
        normalized_date, date_precision = normalize_date(flood_date, pub_date)
        
        # Create example with separate reasoning for location and date
        example = {
            'article_text': text,
            'publication_date': pub_date,
            'primary_location': primary_location,
            'location_confidence': 'high',  # All annotations are high confidence
            'location_reasoning': f"Annotated location: {primary_location}",
            'flood_date': normalized_date,
            'date_precision': date_precision,  # 'month' or 'year' only
            'date_confidence': 'high' if date_precision == 'month' else 'medium',
            'date_reasoning': f"Annotated date: {flood_date}"
        }
        
        training_examples.append(example)
    
    logger.info(f"\nCreated {len(training_examples)} training examples with both location and date")
    
    # Add examples with just location
    location_only = [a for a in with_location if a not in with_both]
    for article in location_only[:30]:  # Limit to 30 additional
        location = article.get('location', '').strip()
        pub_date = article.get('publication_date', 'unknown')
        text = article.get('article_text', '')
        
        primary_location = location.split(';')[0].strip() if ';' in location else location
        
        example = {
            'article_text': text,
            'publication_date': pub_date,
            'primary_location': primary_location,
            'location_confidence': 'high',
            'location_reasoning': f"Annotated location: {primary_location}",
            'flood_date': 'unknown',
            'date_precision': 'unknown',
            'date_confidence': 'low',
            'date_reasoning': "Date not annotated in this article"
        }
        
        training_examples.append(example)
    
    logger.info(f"Added {len(training_examples) - len(with_both)} location-only examples")
    logger.info(f"Total training examples: {len(training_examples)}")
    
    # Split into train and dev
    split_idx = int(len(training_examples) * train_split)
    train_examples = training_examples[:split_idx]
    dev_examples = training_examples[split_idx:]
    
    logger.info(f"\nSplit: {len(train_examples)} training, {len(dev_examples)} development")
    
    # Save
    logger.info(f"\nSaving training data to {output_train}")
    save_json(train_examples, output_train)
    
    logger.info(f"Saving development data to {output_dev}")
    save_json(dev_examples, output_dev)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING DATA PREPARATION COMPLETE")
    logger.info("=" * 70)
    
    # Print some statistics
    logger.info("\nSample training example:")
    if train_examples:
        sample = train_examples[0]
        logger.info(f"  Publication date: {sample['publication_date']}")
        logger.info(f"  Primary location: {sample['primary_location']}")
        logger.info(f"  Flood date: {sample['flood_date']} (precision: {sample['date_precision']})")
        logger.info(f"  Location confidence: {sample['location_confidence']}")
        logger.info(f"  Date confidence: {sample['date_confidence']}")
        logger.info(f"  Text snippet: {sample['article_text'][:200]}...")


def main():
    """Main function"""
    # Set up paths
    input_file = PROJECT_ROOT / 'data' / 'all_labeled_data.json'
    output_dir = PROJECT_ROOT / 'stage3' / 'data'
    output_dir.mkdir(exist_ok=True, parents=True)
    
    output_train = output_dir / 'train_examples.json'
    output_dev = output_dir / 'dev_examples.json'
    
    # Prepare data
    prepare_training_data(input_file, output_train, output_dev)


if __name__ == '__main__':
    main()
