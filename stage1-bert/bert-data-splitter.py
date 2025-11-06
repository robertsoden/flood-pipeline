"""
Data splitting script for BERT training.
Splits labeled data into train/test sets with extraction reserve.

Usage:
    python stage1-bert/scripts/split_data.py
"""
import sys
from pathlib import Path
import json
import random
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent  # Up from stage1-bert/scripts/
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared config
from shared import train_filepath, test_filepath, extraction_filepath

# Configuration
INPUT_FILE = PROJECT_ROOT / 'data' / 'all_labeled_data.json'
EXTRACTION_RESERVE_SIZE = 100
TRAIN_TEST_SPLIT = 0.70  # 70% train, 30% test
RANDOM_SEED = 42

def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_data(data, filepath):
    """Save data to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Saved: {filepath}")

def print_split_stats(data, label="Dataset"):
    """Print statistics about a data split."""
    floods = [x for x in data if x.get('flood_mentioned', False)]
    non_floods = [x for x in data if not x.get('flood_mentioned', False)]
    
    print(f"{label}: {len(data)} total")
    print(f"  - Floods: {len(floods)} ({len(floods)/len(data):.1%})")
    print(f"  - Non-floods: {len(non_floods)} ({len(non_floods)/len(data):.1%})")

def main():
    """Split labeled data into train/test/extraction sets."""
    
    print("="*60)
    print("BERT Data Splitting")
    print("="*60)
    
    # Load all labeled data
    print(f"\nLoading data from: {INPUT_FILE}")
    all_data = load_data(INPUT_FILE)
    
    # Separate by class
    floods = [x for x in all_data if x.get('flood_mentioned', False)]
    non_floods = [x for x in all_data if not x.get('flood_mentioned', False)]
    
    print(f"\nTotal samples: {len(all_data)}")
    print(f"Floods: {len(floods)} ({len(floods)/len(all_data):.1%})")
    print(f"Non-floods: {len(non_floods)} ({len(non_floods)/len(all_data):.1%})")
    
    # Strategy:
    # 1. Set aside 100 examples for extraction reserve (50 floods, 50 non-floods)
    # 2. From remaining: split 70/30 train/test
    
    # Step 1: Separate extraction reserve
    extraction_size_per_class = EXTRACTION_RESERVE_SIZE // 2
    
    floods_train_pool, floods_extraction = train_test_split(
        floods, 
        test_size=min(extraction_size_per_class, len(floods)//6), 
        random_state=RANDOM_SEED
    )
    
    non_floods_train_pool, non_floods_extraction = train_test_split(
        non_floods, 
        test_size=extraction_size_per_class, 
        random_state=RANDOM_SEED
    )
    
    extraction_reserve = floods_extraction + non_floods_extraction
    random.shuffle(extraction_reserve)
    
    # Step 2: Split remaining data 70/30
    test_ratio = 1.0 - TRAIN_TEST_SPLIT
    
    floods_train, floods_test = train_test_split(
        floods_train_pool, 
        test_size=test_ratio, 
        random_state=RANDOM_SEED
    )
    
    non_floods_train, non_floods_test = train_test_split(
        non_floods_train_pool, 
        test_size=test_ratio, 
        random_state=RANDOM_SEED
    )
    
    # Combine and shuffle
    bert_train = floods_train + non_floods_train
    bert_test = floods_test + non_floods_test
    
    random.seed(RANDOM_SEED)
    random.shuffle(bert_train)
    random.shuffle(bert_test)
    
    # Print statistics
    print("\n" + "="*60)
    print(f"DATA SPLIT ({int(TRAIN_TEST_SPLIT*100)}/{int(test_ratio*100)} train/test)")
    print("="*60)
    
    print("\nTrain set:")
    print_split_stats(bert_train, "  Train")
    
    print("\nTest set:")
    print_split_stats(bert_test, "  Test")
    
    print("\nExtraction reserve:")
    print_split_stats(extraction_reserve, "  Extraction")
    
    # Save splits using paths from shared config
    print("\n" + "="*60)
    print("Saving splits...")
    print("="*60)
    
    save_data(bert_train, train_filepath)
    save_data(bert_test, test_filepath)
    save_data(extraction_reserve, extraction_filepath)
    
    print("\n✓ All splits saved successfully!")
    print(f"\nTest set size: {len(bert_test)} examples")
    print(f"Flood examples in test: {len([x for x in bert_test if x['flood_mentioned']])}")
    print("More reliable recall measurement with larger test set!")

if __name__ == '__main__':
    main()