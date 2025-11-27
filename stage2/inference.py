"""
Stage 2 Inference: Run flood verification AND Ontario filtering
Uses Llama 3.1 8B on Bedrock with optimized demos

Two-step process:
1. flood_mentioned - Is this article about a real flood?
2. is_ontario - Did the flood occur in Ontario? (only for verified floods)
"""

import sys
from pathlib import Path
import json
import dspy
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("="*70)
print("STAGE 2 INFERENCE: FLOOD + ONTARIO VERIFICATION (PARALLEL)")
print("="*70)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = PROJECT_ROOT / 'results' / 'predicted_floods.json'
OUTPUT_FILE = PROJECT_ROOT / 'results' / 'stage2_verified_floods.json'
CHECKPOINT_FILE = PROJECT_ROOT / 'results' / 'stage2_checkpoint.json'

FLOOD_MODEL_FILE = PROJECT_ROOT / 'models' / 'stage2_flood_verified.json'
ONTARIO_MODEL_FILE = PROJECT_ROOT / 'models' / 'stage2_ontario_filter.json'

BATCH_SIZE = 500  # Save checkpoint every N articles
NUM_THREADS = 8   # Parallel threads for API calls

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading BERT-filtered articles...")
with open(INPUT_FILE, 'r') as f:
    articles = json.load(f)
print(f"   Total articles: {len(articles)}")

# Check for existing checkpoint
start_idx = 0
results = []
if CHECKPOINT_FILE.exists():
    print("\n   Found checkpoint, resuming...")
    with open(CHECKPOINT_FILE, 'r') as f:
        checkpoint = json.load(f)
    start_idx = checkpoint['processed']
    results = checkpoint['results']
    print(f"   Resuming from article {start_idx}")
else:
    print("   Starting fresh")

remaining = len(articles) - start_idx
print(f"   Remaining: {remaining} articles")

# ============================================================================
# CONFIGURE MODELS
# ============================================================================

print("\n2. Configuring Llama 3.1 8B on Bedrock...")
lm = dspy.LM(
    'bedrock/us.meta.llama3-1-8b-instruct-v1:0',
    temperature=0.1  # Low for deterministic inference
)
dspy.configure(lm=lm)
print("   LLM configured")

print("\n3. Loading optimized models...")

# Flood verification model
flood_verifier = dspy.ChainOfThought('article_text, title -> reasoning, flood_mentioned')
flood_verifier.load(FLOOD_MODEL_FILE)
flood_demos = flood_verifier.predict.demos if hasattr(flood_verifier.predict, 'demos') else []
print(f"   Flood model: {len(flood_demos)} demos")

# Ontario filter model
ontario_filter = dspy.ChainOfThought('article_text, title -> reasoning, is_ontario')
ontario_filter.load(ONTARIO_MODEL_FILE)
ontario_demos = ontario_filter.predict.demos if hasattr(ontario_filter.predict, 'demos') else []
print(f"   Ontario model: {len(ontario_demos)} demos")

# ============================================================================
# INFERENCE (PARALLEL)
# ============================================================================

print(f"\n4. Running inference on {remaining} articles...")
print(f"   Step 1: Check flood_mentioned for all articles")
print(f"   Step 2: Check is_ontario for verified floods only")
print(f"   Threads: {NUM_THREADS}")
print(f"   Checkpoint interval: every {BATCH_SIZE} articles")
est_hours = remaining / (1.2 * NUM_THREADS) / 3600 * 1.5  # 1.5x for two-step
print(f"   Estimated time: {est_hours:.1f} hours")
print(f"   DEBUG: Will save checkpoints at: 500, 1000, 1500, 2000, ...\n")

# Thread-safe counters
lock = threading.Lock()
flood_count = 0
ontario_count = 0
error_count = 0

def save_checkpoint(processed, results):
    """Save progress to checkpoint file."""
    checkpoint = {
        'processed': processed,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)

def process_article(idx_article):
    """Process a single article through both models."""
    global flood_count, ontario_count, error_count
    idx, article = idx_article
    text = article.get('full_text') or article.get('article_text', '')
    title = article.get('title', '')

    result = {
        **article,
        'stage2': {
            'flood_mentioned': None,
            'flood_reasoning': '',
            'is_ontario': None,
            'ontario_reasoning': '',
            'error': None
        }
    }

    try:
        # Step 1: Check if flood is mentioned
        flood_pred = flood_verifier(article_text=text, title=title)
        flood_mentioned = flood_pred.flood_mentioned

        if isinstance(flood_mentioned, str):
            flood_mentioned = flood_mentioned.lower() == 'true'

        result['stage2']['flood_mentioned'] = flood_mentioned
        result['stage2']['flood_reasoning'] = getattr(flood_pred, 'reasoning', '')

        # Step 2: If flood verified, check if Ontario
        if flood_mentioned:
            ontario_pred = ontario_filter(article_text=text, title=title)
            is_ontario = ontario_pred.is_ontario

            if isinstance(is_ontario, str):
                is_ontario = is_ontario.lower() == 'true'

            result['stage2']['is_ontario'] = is_ontario
            result['stage2']['ontario_reasoning'] = getattr(ontario_pred, 'reasoning', '')

    except Exception as e:
        result['stage2']['error'] = str(e)

    return idx, result

# Prepare work items
work_items = [(i, articles[i]) for i in range(start_idx, len(articles))]

# Process in parallel with progress tracking
results_dict = {i: r for i, r in enumerate(results)}  # Index existing results

with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    futures = {executor.submit(process_article, item): item[0] for item in work_items}

    with tqdm(total=len(work_items), initial=0, desc="Processing") as pbar:
        for future in as_completed(futures):
            try:
                idx, result = future.result()
                results_dict[idx] = result

                with lock:
                    if result['stage2'].get('flood_mentioned'):
                        flood_count += 1
                    if result['stage2'].get('is_ontario'):
                        ontario_count += 1
                    if result['stage2'].get('error'):
                        error_count += 1

                pbar.update(1)

                # Checkpoint periodically
                processed = len(results_dict)
                if processed % BATCH_SIZE == 0:
                    tqdm.write(f"   DEBUG: Checkpoint triggered at {processed} articles")
                    ordered_results = [results_dict[i] for i in sorted(results_dict.keys())]
                    save_checkpoint(processed, ordered_results)
                    tqdm.write(f"   ✓ Checkpoint saved: {processed} done | {flood_count} floods | {ontario_count} Ontario | {error_count} errors")

            except Exception as e:
                # Log the error but continue processing
                idx = futures.get(future, -1)
                tqdm.write(f"   ERROR processing article {idx}: {e}")
                pbar.update(1)
                continue

# Final ordered results
results = [results_dict[i] for i in sorted(results_dict.keys())]

# ============================================================================
# SAVE FINAL RESULTS
# ============================================================================

print("\n5. Saving results...")

# Save all results
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"   Saved: {OUTPUT_FILE}")

# Save only verified floods
verified_floods = [r for r in results if r.get('stage2', {}).get('flood_mentioned')]
verified_output = PROJECT_ROOT / 'results' / 'stage2_floods_only.json'
with open(verified_output, 'w') as f:
    json.dump(verified_floods, f, indent=2)
print(f"   Saved: {verified_output} ({len(verified_floods)} articles)")

# Save only Ontario floods (final output!)
ontario_floods = [r for r in results if r.get('stage2', {}).get('is_ontario')]
ontario_output = PROJECT_ROOT / 'results' / 'stage2_ontario_floods.json'
with open(ontario_output, 'w') as f:
    json.dump(ontario_floods, f, indent=2)
print(f"   Saved: {ontario_output} ({len(ontario_floods)} articles)")

# Clean up checkpoint
if CHECKPOINT_FILE.exists():
    CHECKPOINT_FILE.unlink()
    print("   Removed checkpoint file")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"   Input articles:     {len(articles)}")
print(f"   Verified floods:    {len(verified_floods)} ({len(verified_floods)/len(articles)*100:.1f}%)")
print(f"   Ontario floods:     {len(ontario_floods)} ({len(ontario_floods)/len(articles)*100:.1f}%)")
print(f"   Errors:             {error_count}")
print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*70 + "\n")
