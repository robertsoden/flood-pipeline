# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-stage NLP pipeline to identify and extract flood information from 50,000+ Ontario newspaper articles using BERT filtering followed by LLM-based verification and extraction.

**Pipeline Stages:**
- **Stage 1 (BERT):** High-recall filter (>95%) to exclude non-flood articles
- **Stage 2 (Local LLM):** High-precision flood verification + Ontario location check
- **Stage 3 (Local LLM + Geocoding):** Extract flood date/location, geocode to lat/lon
- **Stage 4 (External LLM):** Extract detailed flood impacts

**Current Status:** Stage 1 (BERT) is complete. Stages 2-4 are in development.

## Architecture

### Data Flow
Articles are progressively enriched through each stage:
```
Raw articles (50K)
  → Stage 1: Filter to ~5-15K candidates (BERT)
  → Stage 2: Verify ~2-8K confirmed Ontario floods (LLM)
  → Stage 3: Geolocate floods with dates
  → Stage 4: Extract detailed impacts
```

Each stage adds a `stageN` key to article JSON with its results.

### Directory Structure
- `shared/` - Shared configuration and utilities used across all stages
  - `config.py` - **Central configuration for all paths, models, and settings**
  - `utils.py` - Shared functions (DSPy data prep, JSON I/O, etc.)
- `stage1-bert/` - BERT-based filtering (complete)
  - `bert-train.py` - Semi-supervised BERT training with pseudo-labeling
  - `bert-inference.py` - Apply trained model to full dataset
  - `bert-data-splitter.py` - Split labeled data into train/test
  - `data/` - Training/test datasets
- `stage2/` - LLM flood verification (in progress)
- `stage3/` - Location/date extraction (to do)
- `stage4/` - Impact extraction (to do)
- `working/` - Experimental/optimization scripts (DSPy-based)
  - `pipeline.py` - DSPy pipeline with Phoenix tracing
  - `optimize_*.py` - DSPy optimization experiments
  - `metrics.py` - Evaluation metrics for DSPy
  - `signatures.py` - DSPy signatures for tasks
- `data/` - Raw articles (not in git, requires archive access)
- `models/` - Trained BERT models (not in git)
- `results/` - Pipeline outputs (not in git)
- `annotations/` - Manual labeling tool

## Key Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment (copy and edit if needed)
cp env.example .env

# Note: Requires access to newspaper article archive (not in git)
```

### Stage 1 (BERT Training & Inference)
```bash
# Train BERT model with semi-supervised learning
python stage1-bert/bert-train.py

# Run inference on full dataset
python stage1-bert/bert-inference.py

# Split labeled data for training (if needed)
python stage1-bert/bert-data-splitter.py
```

### Development/Experimentation (DSPy)
```bash
# Run full DSPy pipeline with Phoenix tracing
python working/pipeline.py

# Stage-specific optimization experiments
python working/optimize_stage1_flood.py
python working/optimize_stage1_5_ontario.py
python working/optimize_stage2_details.py
```

## Important Configuration Details

### Centralized Configuration
**All configuration lives in `shared/config.py`** - paths, model settings, stage parameters. Do not hardcode paths elsewhere.

Key config sections:
- `PROJECT_ROOT` - Computed from config file location
- Data file paths (train/test/unlabeled)
- `MODEL_CONFIG` - Ollama LLM settings for DSPy
- `STAGE1_CONFIG` - BERT model path, hyperparameters, threshold
- `STAGE2_CONFIG` - DSPy optimization settings

### Data Format
Articles use flat JSON format with these key fields:
- `full_text` / `article_text` - Article content
- `publication_date` / `date` - Publication date
- `flood_mentioned` - Boolean (labeled examples)
- `location` - Flood location (if applicable)
- `flood_date` - Date of flood event
- `is_ontario` - Boolean for Ontario floods
- `impacts` - Flood impact details

DSPy examples are created using `shared/utils.py:prepare_data()` which handles both nested annotations format and flat format.

### Stage 1 BERT Model
- Uses semi-supervised learning with pseudo-labeling
- Trains for multiple iterations, adding high-confidence predictions
- Optimized for **high recall (>95%)** to avoid missing flood articles
- Class weights favor recall over precision
- Uses F1 (not recall) to select best epoch to avoid overfitting
- Threshold is tuned post-training to achieve target recall
- Model and threshold info saved to `models/balanced_high_recall_iter{N}/`

### DSPy Integration
- LLM backend: Ollama with Qwen2.5:14b-instruct (local)
- Uses BootstrapFewShotWithRandomSearch for optimization
- Phoenix tracing enabled in `working/pipeline.py`
- Metrics defined in `working/metrics.py`
- Task signatures in `working/signatures.py`

## Development Notes

### Python Path Management
Stage scripts add PROJECT_ROOT to sys.path for importing shared modules:
```python
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from shared.config import ...
```

### Model Files
BERT models are large and stored locally in `models/`. Git ignores this directory. The active model is specified in `shared/config.py` via `STAGE1_CONFIG['bert_model_dir']`.

### Data Access
Raw newspaper articles are not in git. The pipeline expects `data/articles_restructured.json` with 50K+ articles. See DATA_README.md for source information.

### Working Directory
The `working/` folder contains experimental DSPy-based implementations. These scripts explore optimization strategies and are not part of the production pipeline yet.
