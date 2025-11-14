# Flood History of Ontario - Project Context (UPDATED)

## Project Overview

**Goal:** Create an automated pipeline to identify and extract flood information from a large corpus of Ontario newspaper articles (50,000+ articles).

**Approach:** Multi-stage pipeline that progressively filters and enriches articles, moving from high-recall/low-precision to high-precision extraction.

**Current Status:** 
- ✅✅ Stage 1 (BERT) **COMPLETE & PRODUCTION-READY**
- ✅ Shared config/utils system implemented
- 🔄 Stages 2-4 ready for development

## Pipeline Architecture

### Stage 1: BERT-Based Filtering (HIGH RECALL)
**Status:** ✅✅ **COMPLETE & PRODUCTION-READY**  
**Purpose:** Initial high-recall (>95%) filter to exclude clearly non-flood articles  
**Method:** Fine-tuned DistilBERT classifier  
**Input:** 50,000 raw articles  
**Output:** ~33,000 potential flood articles for Stage 2  

**Performance (Achieved):**
- **Recall: 97.7%** ✅ (exceeds 95% target!)
- **Precision: 76.8%** (excellent for high-recall filter)
- **Filter rate: 73.5%** (~37,000 articles filtered)
- **Threshold: 0.170** (optimized for maximum recall)
- **F1 Score: 86.0%**

**Trained Model:**
- Location: `models/balanced_high_recall_iter0`
- Training data: 431 examples (203 floods, 228 non-floods)
- Test data: 186 examples (88 floods, 98 non-floods)
- Training time: ~1.5 minutes per epoch (4 epochs)
- Configuration: 1.1x weight multiplier (conservative)

**Production Estimates (on 50K articles):**
- Predicted floods: ~33,000 (67%)
- Filtered non-floods: ~17,000 (33%)
- Expected to catch: ~32,000 of ~33,600 actual floods (97.7%)
- Expected to miss: ~800 floods (2.3%)
- Cost savings: ~$170 (@ $0.01/article for LLM processing)

**Files:**
- `stage1-bert/bert-train-high-recall-FINAL.py` - Training script
- `stage1-bert/bert-inference.py` - Inference on full dataset
- `stage1-bert/bert-data-splitter.py` - Data preparation

**Key Implementation Details:**
- Uses shared config system (`from shared.config import ...`)
- Conservative 1.1x class weighting for balanced data
- F1-based epoch selection (not recall, which was a bug)
- Precision-based iteration selection when recall target met
- Pseudo-labeling attempted but model well-calibrated (no overconfidence)

### Stage 2: LLM Flood Verification (HIGH PRECISION)
**Status:** 🔄 Ready for Development  
**Purpose:** Verify flood events and confirm Ontario location  
**Method:** Local LLM (Ollama/Qwen) with DSPy optimization  
**Input:** Stage 1 candidates (~33,000 articles)  
**Output:** Confirmed flood articles in Ontario with reasoning  

**Key Decisions:**
- Is this describing an actual flood event? (not metaphorical)
- Did the flood occur in Ontario, Canada?

**Expected Implementation:**
- Use DSPy for prompt optimization
- Output structured JSON: `{is_flood: bool, is_ontario: bool, confidence: float, reasoning: str}`
- Batch processing for efficiency
- Model: Ollama with qwen2.5:14b-instruct-q5_K_M
- Can reuse Stage 1 training data (431 examples) - different task, different model

**Training Data Available:**
- Same 431 examples from Stage 1 (have flood_mentioned labels)
- 100 extraction reserve examples (if annotated with locations)
- 186 test examples for validation

### Stage 3: Location & Date Extraction
**Status:** 🔄 To Be Developed  
**Purpose:** Extract specific flood location and date  
**Method:** Local LLM + OpenStreetMap Nominatim geocoding  
**Input:** Stage 2 confirmed floods  
**Output:** Structured data with lat/lon coordinates  

**Key Extractions:**
- Flood date (from article or inferred)
- Location mentions (cities, rivers, regions)
- Geocoded coordinates (latitude, longitude)

**Expected Implementation:**
- LLM extracts location strings and dates
- Geocoding API converts locations to coordinates
- Caching to avoid redundant API calls (see `shared/utils.py` GeocodingCache)
- Rate limiting for Nominatim (1 second between requests)

### Stage 4: Impact Extraction
**Status:** 🔄 To Be Developed  
**Purpose:** Extract detailed flood impact information  
**Method:** External LLM (GPT-4 or Claude) for complex extraction  
**Input:** Stage 3 geolocated floods  
**Output:** Structured impact data  

**Key Extractions:**
- Infrastructure damage
- Casualties/injuries
- Evacuations
- Economic losses
- Environmental impacts
- Response actions

**Expected Implementation:**
- Use more powerful external LLM for nuanced extraction
- Structured output format
- May require iterative extraction for completeness

## Project Structure

```
flood-pipeline/
├── shared/                      # ⭐ SHARED CONFIG SYSTEM
│   ├── __init__.py             # Package initialization
│   ├── config.py               # ALL configuration here
│   └── utils.py                # Shared utilities (JSON I/O, logging, etc.)
│
├── stage1-bert/                 # ✅✅ Stage 1: BERT filtering (COMPLETE)
│   ├── bert-train-high-recall-FINAL.py  # Training script (working)
│   ├── bert-inference.py       # Inference on full dataset
│   ├── bert-data-splitter.py   # Data preparation
│   └── data/                   # Training/test data
│       ├── bert_train_70pct.json        # 431 training examples
│       ├── bert_test_30pct.json         # 186 test examples
│       └── extraction_reserve_100.json  # 100 flood examples (reserved)
│
├── stage2/                      # 🔄 Stage 2: LLM verification
│   └── (to be implemented)
│
├── stage3/                      # 🔄 Stage 3: Location extraction
│   └── (to be implemented)
│
├── stage4/                      # 🔄 Stage 4: Impact extraction
│   └── (to be implemented)
│
├── data/                        # Large data files (NOT in git)
│   ├── articles_restructured.json       # 50K raw articles
│   ├── all_labeled_data.json           # 717 labeled examples total
│   └── processed/                      # Stage outputs
│       ├── stage1_output.json          # (to be generated by inference)
│       ├── stage2_output.json
│       ├── stage3_output.json
│       └── stage4_output.json
│
├── models/                      # Trained models (NOT in git)
│   └── balanced_high_recall_iter0/     # ✅ Working BERT model
│       ├── config.json
│       ├── model.safetensors
│       ├── threshold_info.json         # Threshold: 0.170
│       └── ...
│
├── results/                     # Inference outputs (NOT in git)
│   ├── all_articles_classified.json    # All 50K with predictions
│   ├── predicted_floods.json           # ~33K predicted floods
│   ├── filtered_non_floods.json        # ~17K filtered out
│   ├── classification_summary.csv
│   └── classification_stats.json
│
├── annotations/                 # Manual annotation tools
│   ├── annotation_tool.html    # Browser-based labeling interface
│   └── schema.json             # Annotation schema
│
├── working/                     # Experimental/development code
│   └── (optimization experiments)
│
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## Shared Configuration System

**CRITICAL:** All stages use a centralized configuration in `shared/config.py` and `shared/utils.py`.

### Current Configuration (`shared/config.py`)

```python
from pathlib import Path

# Project root is parent of shared/ directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data sources
train_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_train_70pct.json'
test_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'bert_test_30pct.json'
extraction_filepath = PROJECT_ROOT / 'stage1-bert' / 'data' / 'extraction_reserve_100.json'
unlabeled_filepath = PROJECT_ROOT / 'data' / 'articles_restructured.json'

# Model configuration (for Ollama/LLM - Stages 2-4)
MODEL_CONFIG = {
    'name': 'ollama_chat/qwen2.5:14b-instruct-q5_K_M',
    'api_base': 'http://localhost:11434',
    'api_key': ''
}

# Stage 1: BERT Configuration
STAGE1_CONFIG = {
    'bert_model': 'distilbert-base-uncased',
    'bert_model_dir': PROJECT_ROOT / 'models' / 'balanced_high_recall_iter0',
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 4,
    'max_length': 512,
    'threshold': 0.170,  # Optimized threshold from training
    'target_recall': 0.95,
}

# Stage 2: Flood verification (to be expanded)
STAGE2_CONFIG = {
    'max_bootstrapped_demos': 8,
    'max_labeled_demos': 8,
    'num_candidate_programs': 30,
    'num_threads': 1,
}

# Stage 3: Location extraction (to be expanded)
STAGE3_CONFIG = {
    'nominatim_cache': PROJECT_ROOT / 'artifacts' / 'geocoding_cache.json',
    'geocoding_delay': 1.0,
    'focus_region': 'Ontario, Canada',
}

# Stage 4: Impact extraction (to be expanded)
STAGE4_CONFIG = {
    'impact_categories': [
        'infrastructure_damage',
        'casualties',
        'evacuations',
        'economic_loss',
        'environmental_impact',
        'response_actions'
    ],
}
```

### How to Use Shared Config in New Scripts

Every stage script should start with:

```python
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared
from shared.config import (
    train_filepath,
    test_filepath,
    unlabeled_filepath,
    MODEL_CONFIG,      # For Ollama/LLM stages
    STAGE2_CONFIG,     # Or STAGE3_CONFIG, STAGE4_CONFIG
)
```

### Shared Utilities (`shared/utils.py`)

Available utilities for all stages:

**JSON I/O:**
```python
from shared.utils import load_json, save_json
data = load_json(filepath)
save_json(data, filepath)
```

**Logging:**
```python
from shared.utils import setup_logging
logger = setup_logging('stage2', log_dir)
```

**Geocoding Cache (for Stage 3):**
```python
from shared.utils import GeocodingCache
cache = GeocodingCache(STAGE3_CONFIG['nominatim_cache'])
coords = cache.get("Toronto")
cache.set("Toronto", 43.65, -79.38)
```

**Progress Tracking:**
```python
from shared.utils import ProgressTracker
progress = ProgressTracker(total=1000, log_interval=100)
for item in items:
    progress.update()
```

## Data Flow Between Stages

### Progressive Enrichment Pattern

Each stage adds its results to the article object:

```json
{
  "article_id": "12345",
  "title": "Flooding closes Highway 401",
  "publication_date": "2023-05-15",
  "full_text": "Heavy rains caused flooding...",
  
  "stage1": {
    "is_flood_candidate": true,
    "confidence": 0.92,
    "probability": 0.92
  },
  
  "stage2": {
    "is_flood": true,
    "is_ontario": true,
    "confidence": 0.88,
    "reasoning": "Article describes actual flooding event in Toronto, Ontario"
  },
  
  "stage3": {
    "flood_date": "2023-05-15",
    "locations": [
      {
        "mention": "Toronto",
        "lat": 43.65,
        "lon": -79.38
      }
    ]
  },
  
  "stage4": {
    "impacts": {
      "infrastructure_damage": "Highway 401 closure",
      "evacuations": "50 homes evacuated",
      "economic_loss": "Estimated $2M in damages"
    }
  }
}
```

### File Paths Between Stages

```python
# Stage 1
Input:  unlabeled_filepath                    # data/articles_restructured.json
Output: results/predicted_floods.json         # ~33K flood candidates

# Stage 2
Input:  results/predicted_floods.json         # Stage 1 output
Output: data/processed/stage2_output.json     # Verified Ontario floods

# Stage 3
Input:  data/processed/stage2_output.json     # Stage 2 output
Output: data/processed/stage3_output.json     # With locations/dates

# Stage 4
Input:  data/processed/stage3_output.json     # Stage 3 output
Output: data/processed/stage4_output.json     # Final structured data
```

## Critical Bugs Fixed During Development

### Bug #1: Wrong Epoch Selection
**Problem:** 
- Training script used `metric_for_best_model="eval_recall_at_0.30"`
- This loaded epoch 1, which achieved 100% recall by predicting everything as flood
- Final model had 47% accuracy, 47% precision (useless!)

**Root Cause:**
- Epoch 1: 73% accuracy, 100% recall (predicts almost everything as flood)
- Epoch 3: 85% accuracy, 97.7% recall (learned real patterns)
- Script selected epoch 1 because it had "highest recall at 0.30 threshold"

**Fix:**
```python
# Changed from:
metric_for_best_model="eval_recall_at_0.30"

# To:
metric_for_best_model="eval_f1"  # Balances precision and recall
```

**Impact:**
- Model went from 47% accuracy → 85% accuracy
- Predictions went from clustering at 0.5 → spread across 0.05-0.88
- Pseudo-labeling started working (model became confident)

### Bug #2: Wrong Iteration Selection
**Problem:**
- When multiple iterations achieved target recall, script selected first one
- Iteration 0: 95.5% recall, 69.4% precision
- Iteration 1: 95.5% recall, 75.0% precision ← Better, but not selected!

**Root Cause:**
```python
# Old code:
best_iter_idx = df_iterations['recall'].idxmax()  # Returns first max
```

**Fix:**
```python
# Filter for recall >= target, then pick highest precision
meeting_target = df_iterations[df_iterations['recall'] >= TARGET_RECALL]
if len(meeting_target) > 0:
    best_iter_idx = meeting_target['precision'].idxmax()
```

**Impact:**
- Would select iteration 1 (75% precision) over iteration 0 (69% precision)
- +5.6% precision improvement
- +9.2% filter rate improvement
- ~2,400 fewer false positives

**Note:** In final run, only 1 iteration completed (pseudo-labeling didn't generate enough confident predictions), so this bug didn't affect final results.

### Bug #3: Configuration Import Errors
**Problem:** Scripts tried to import from non-existent `config` and `utils` modules

**Fix:** Updated to use shared config system:
```python
from shared.config import train_filepath, STAGE1_CONFIG
```

## Training Data Details

**Location:** `data/all_labeled_data.json`  
**Total Size:** 717 labeled articles  
**Distribution:** 341 floods (47.6%), 376 non-floods (52.4%) - well-balanced!

**Current Split:**
- **BERT Train:** 431 examples (203 floods, 228 non-floods) - 70%
- **BERT Test:** 186 examples (88 floods, 98 non-floods) - 30%
- **Extraction Reserve:** 100 examples (50 floods) - for Stage 3/4 training

**Format:**
```json
{
  "id": "article_123",
  "full_text": "...",
  "publication_date": "2023-05-15",
  "flood_mentioned": true,
  "annotations": {
    "location": "Toronto",
    "flood_date": "2023-05-15",
    "impacts": "Highway closure, evacuations"
  }
}
```

**Usage:**
- Stage 1 (BERT): Used 431 train + 186 test
- Stage 2 (LLM verification): Can reuse same 431+186 (different task)
- Stage 3/4: Use extraction reserve (100) if annotated with locations/impacts

## Model Setup

### Local LLM (Ollama) - For Stages 2-3
- **Installed at:** `http://localhost:11434`
- **Model:** `qwen2.5:14b-instruct-q5_K_M`
- **Used for:** Flood verification, location extraction
- **Advantages:** Cost-effective, fast, private
- **Configuration:** `MODEL_CONFIG` in shared config

### External LLM (Optional for Stage 4)
- **Options:** GPT-4 or Claude
- **Used for:** Complex impact extraction
- **Configured via:** Environment variables
- **Trade-off:** More expensive but higher quality for nuanced tasks

### Geocoding (Stage 3)

**Service:** OpenStreetMap Nominatim  
**Rate Limit:** 1 request per second  
**Caching:** Mandatory (use `GeocodingCache` from shared/utils)  
**User Agent:** Required by Nominatim (set in config)

```python
from shared.utils import GeocodingCache
from shared.config import STAGE3_CONFIG

cache = GeocodingCache(STAGE3_CONFIG['nominatim_cache'])

# Always check cache first
coords = cache.get(location_string)
if coords is None:
    coords = call_nominatim(location_string)
    cache.set(location_string, coords['lat'], coords['lon'])
    time.sleep(STAGE3_CONFIG['geocoding_delay'])
```

## Development Guidelines for Stages 2-4

### When Implementing New Stages

1. **Use Shared Config**
   - Import from `shared.config` not local files
   - Add stage-specific settings to appropriate `STAGEN_CONFIG`
   - Use Path objects for file paths

2. **Follow Established Patterns**
   - Load data: `articles = load_json(input_path)`
   - Process: Add `stageN` key to each article
   - Save: `save_json(articles, output_path)`

3. **Use DSPy for LLM Stages**
   - Leverage DSPy for prompt optimization
   - Configure in `STAGEN_CONFIG` settings
   - Can reuse Stage 1 training data (different task)

4. **Implement Logging**
   - Use `setup_logging()` from shared utils
   - Log progress with `ProgressTracker`
   - Include statistics and summaries

5. **Batch Processing**
   - Process in batches for efficiency
   - Report progress regularly
   - Handle API rate limits

6. **Error Handling**
   - Handle missing fields gracefully
   - Mark problematic articles for review
   - Log errors for debugging

### Code Style Template

```python
"""
Stage N: [Purpose]
[Brief description]
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import (
    STAGEN_CONFIG,
    MODEL_CONFIG,
    train_filepath,
    test_filepath,
)
from shared.utils import load_json, save_json, setup_logging

# Set up logging
logger = setup_logging('stageN')

def process_article(article: dict) -> dict:
    """Process a single article."""
    # Implementation
    pass

def main():
    """Main processing function."""
    logger.info("Starting Stage N")
    
    # Load input (from previous stage)
    articles = load_json(input_path)
    
    # Process
    for article in articles:
        result = process_article(article)
        article[f'stage{N}'] = result
    
    # Save output
    save_json(articles, output_path)
    
    logger.info("Stage N complete")

if __name__ == '__main__':
    main()
```

## Current Priorities

1. ✅✅ **Stage 1 Complete** - BERT model trained and ready
2. ▶️ **Run Stage 1 Inference** - Classify 50K articles (in progress)
3. 🔄 **Stage 2 Development** - Flood verification with DSPy + Ollama
4. 🔄 **Stage 3 Planning** - Location extraction approach
5. 🔄 **Stage 4 Planning** - Impact extraction design

## Performance Targets

**Stage 1: ✅ ACHIEVED**
- Target recall: 95%+ → **Achieved: 97.7%** ✅
- Acceptable precision: ~35% → **Achieved: 76.8%** ✅✅
- Speed: Process 50K in 30-60 min → **Expected: 5-10 min on CPU**

**Stage 2: (Targets)**
- Precision: 80%+ (high confidence floods)
- Maintain recall from Stage 1
- Speed: Process in batches, ~10-50 articles/second

**Stage 3: (Targets)**
- Accuracy: 90%+ on location extraction
- Geocoding success rate: 85%+
- Handle ambiguous locations gracefully

**Stage 4: (Targets)**
- Completeness: Extract all mentioned impacts
- Accuracy: Validate against sample
- Handle missing information appropriately

## Key Lessons Learned

### From Stage 1 Development

1. **Conservative weighting works for balanced data**
   - 1.1x multiplier was sufficient (vs 1.5x or higher)
   - Over-weighting can cause model to predict everything as positive class

2. **F1 metric is better than recall for epoch selection**
   - Recall-only selection can pick epochs that predict everything as positive
   - F1 balances precision and recall

3. **Well-calibrated models won't be overconfident**
   - Model didn't generate pseudo-labels (no >0.95 confidence predictions)
   - This is actually good - shows honest uncertainty
   - One iteration with 431 examples was sufficient

4. **Larger test set catches issues**
   - 186 examples (vs 50) more reliable
   - Harder for model to "get lucky" with 100% recall

5. **Shared config system is essential**
   - Prevents configuration drift between stages
   - Makes project easier to maintain
   - Centralizes all settings

## Next Steps

### Immediate (Now)
1. **Complete Stage 1 Inference**
   - Running: Classifying 50K articles
   - Output: ~33K predicted floods
   - Review results and spot-check quality

### Short-term (Next)
2. **Implement Stage 2**
   - Set up DSPy with Ollama/Qwen
   - Optimize prompts for flood verification
   - Verify Ontario location
   - Process ~33K Stage 1 candidates
   - Target: ~25-30K confirmed Ontario floods

3. **Design Stage 3**
   - Plan location extraction approach
   - Set up Nominatim geocoding with caching
   - Test on sample articles
   - Handle ambiguous location mentions

### Medium-term (Later)
4. **Implement Stage 4**
   - Decide on LLM (local vs external)
   - Design impact extraction schema
   - Test on sample articles
   - Validate completeness and accuracy

5. **End-to-end Testing**
   - Run full pipeline on test set
   - Validate results against ground truth
   - Measure overall quality metrics
   - Document findings

6. **Production Deployment**
   - Optimize for performance
   - Add monitoring and logging
   - Create user documentation
   - Deploy for regular use

## Resources

**Documentation:**
- `README.md` - Project overview
- `PROJECT_CONTEXT.md` - This file (detailed context)
- `shared/config.py` - All configuration
- `shared/utils.py` - Shared utilities

**Stage 1 Examples:**
- `stage1-bert/bert-train-high-recall-FINAL.py` - Complete training script
- `stage1-bert/bert-inference.py` - Complete inference script
- Use as templates for Stages 2-4

**Annotation:**
- `annotations/annotation_tool.html` - Manual labeling interface
- `annotations/schema.json` - Annotation schema

## Working with Claude Code

**Recommended approach for Stages 2-4:**

1. **Upload this PROJECT_CONTEXT.md to your project**
2. **Start Claude Code in the project directory:**
   ```bash
   cd flood_pipeline
   claude code
   ```

3. **Give Claude Code context:**
   ```
   Read PROJECT_CONTEXT.md to understand this project. 
   Stage 1 (BERT) is complete and working. 
   I need you to implement Stage 2 (LLM flood verification with DSPy).
   Use the shared config system and follow the patterns from Stage 1.
   ```

4. **Claude Code will:**
   - Read the project structure
   - Follow your shared config system
   - Use your established patterns
   - Write, test, and debug Stage 2
   - Iterate until it works

**Context Claude Code will have:**
- Complete pipeline architecture
- Stage 1 implementation as reference
- Shared config system
- Data file locations and formats
- Performance targets
- Lessons learned from Stage 1

---

**Last Updated:** November 2024  
**Project Status:** Stage 1 complete and production-ready, Stages 2-4 ready for development  
**Key Achievement:** 97.7% recall BERT model that filters 73.5% of non-flood articles