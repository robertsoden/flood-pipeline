# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Purpose:** Extract flood event information from 50,000+ Ontario newspaper articles using a 3-stage NLP pipeline.

**Output:** Geocoded flood articles with dates and locations, ready for integration with the `flood_triangulation` system.

**Status:** Pipeline complete. All 3 stages operational.

## Pipeline Architecture

```
Raw Articles (50,247)
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: BERT Classification                                     │
│ - Semi-supervised learning with pseudo-labeling                  │
│ - High-recall filter (>95%) to avoid missing floods              │
│ Output: 5,247 flood candidate articles                           │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: LLM Flood Verification                                  │
│ - Confirms actual flood events (not just mentions)               │
│ - Verifies Ontario location                                      │
│ Output: 2,290 verified Ontario flood articles                    │
└─────────────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: Information Extraction & Geocoding                      │
│ - NER extraction of locations/dates (spaCy)                      │
│ - LLM verification and correction (Claude Sonnet)                │
│ - Mapbox geocoding to lat/lon                                    │
│ Output: 1,940 geocoded articles with dates/locations             │
└─────────────────────────────────────────────────────────────────┘
       ↓
    Export to flood_triangulation for cataloguing with database sources
```

## Directory Structure

```
flood_news/
├── shared/                 # Shared configuration and utilities
│   ├── config.py          # Central configuration (paths, models, settings)
│   ├── utils.py           # JSON I/O, data prep utilities
│   ├── validation.py      # Pydantic schemas, article ID generation
│   ├── checkpoint.py      # Checkpoint/resume functionality
│   ├── dspy_utils.py      # DSPy LM configuration helpers
│   └── logging_config.py  # Standardized logging setup
│
├── stage1-bert/           # BERT-based filtering (✅ Complete)
│   ├── bert-train.py      # Semi-supervised training with pseudo-labeling
│   ├── bert-inference.py  # Apply trained model to full dataset
│   └── data/              # Training/test datasets
│
├── stage2/                # LLM flood verification (✅ Complete)
│   ├── process.py         # Main processing script (supports --resume)
│   ├── optimize.py        # DSPy optimization for model training
│   ├── signatures.py      # DSPy signatures for verification
│   └── metrics.py         # Evaluation metrics
│
├── stage3/                # Location/date extraction (✅ Complete)
│   ├── process_ner.py     # NER extraction with spaCy
│   ├── process_llm_verify.py  # LLM verification (supports --resume)
│   ├── ner_extractor.py   # NER extraction logic with 300+ Ontario places
│   ├── geocode.py         # Mapbox geocoding (caches results)
│   └── signatures.py      # DSPy signatures for extraction
│
├── tests/                 # Test suite
│   ├── test_integration.py    # Pipeline data flow tests
│   ├── test_extraction.py     # NER extraction accuracy tests
│   ├── test_edge_cases.py     # Edge case handling tests
│   └── conftest.py            # Pytest fixtures
│
├── results/               # Pipeline outputs (not in git)
│   ├── stage2_ontario_floods.json
│   ├── stage3_verified.json
│   └── stage3_geocoded.json  # Final output
│
├── checkpoints/           # Resume checkpoints (not in git)
├── models/                # Trained BERT models (not in git)
├── data/                  # Raw articles (not in git)
└── logs/                  # Processing logs
```

## Key Commands

### Full Pipeline Run

```bash
# Stage 1: BERT classification
python stage1-bert/bert-inference.py

# Stage 2: LLM flood verification
python stage2/process.py

# Stage 3: NER + LLM verification + Geocoding
python stage3/process_ner.py           # Extract locations/dates with NER
python stage3/process_llm_verify.py    # Verify/correct with LLM
python stage3/geocode.py               # Geocode to lat/lon
```

### Resume After Interruption

Stages 2 and 3 support checkpoint/resume for long-running processing:

```bash
# Resume Stage 2 from last checkpoint
python stage2/process.py --resume

# Resume Stage 3 LLM verification from checkpoint
python stage3/process_llm_verify.py --resume

# Start fresh (clear existing checkpoints)
python stage2/process.py --clear-checkpoint
python stage3/process_llm_verify.py --clear-checkpoint

# Adjust checkpoint frequency (default: 100 for stage2, 50 for stage3)
python stage2/process.py --checkpoint-interval 50
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=shared --cov=stage3 -v
```

### Environment Setup

```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model for NER
python -m spacy download en_core_web_sm

# Configure API keys
cp env.example .env
# Edit .env with MAPBOX_TOKEN and ANTHROPIC_API_KEY
```

## Current Results

| Stage | Input | Output | Rate |
|-------|-------|--------|------|
| Stage 1 (BERT) | 50,247 articles | 5,247 candidates | 10.4% |
| Stage 2 (LLM) | 5,247 candidates | 2,290 Ontario floods | 43.6% |
| Stage 3 (Geocode) | 2,290 articles | 1,940 geocoded | 84.7% |

### Stage 3 Quality Metrics

- Location verified (NER correct): 35.5%
- Location corrected by LLM: 64.5%
- Date high confidence: 49.9%
- Date medium confidence: 16.5%
- Date low/not found: 33.6%

## Configuration

All configuration in `shared/config.py`:

- `PROJECT_ROOT` - Base path for all file operations
- `STAGE1_CONFIG` - BERT model path, threshold (0.170 for high recall)
- `STAGE2_CONFIG` - LLM settings for flood verification
- `STAGE3_CONFIG` - NER model, geocoding settings

### Environment Variables (.env)

```
MAPBOX_TOKEN=pk.xxx          # For geocoding
ANTHROPIC_API_KEY=sk-xxx     # For Claude LLM calls
```

## Data Format

Articles are progressively enriched through each stage:

```json
{
  "article_id": "12345",
  "full_text": "...",
  "publication_date": "2013-07-08",
  "stage1": {
    "flood_probability": 0.94,
    "is_flood": true
  },
  "stage2": {
    "is_flood": true,
    "is_ontario": true,
    "confidence": "high"
  },
  "stage3": {
    "location": "Toronto",
    "flood_date": "July 2013",
    "latitude": 43.6532,
    "longitude": -79.3832,
    "date_confidence": "high"
  }
}
```

## Integration with flood_triangulation

This repo's output (`results/stage3_geocoded.json`) feeds into the `flood_triangulation` system:

```
flood_news/results/stage3_geocoded.json
        ↓
flood_triangulation/scripts/pipeline/ingest/ingest_article_cases.py
        ↓
Flood inventory combining articles + database records + hydro data
```

In flood_triangulation:
1. Articles become **cases** in the unified schema
2. Cases cluster into **events** with database records
3. Events are enriched with hydrometric evidence
4. Cross-referencing identifies confirmed floods and gaps

This repo focuses on NLP extraction; triangulation and cataloguing happen downstream.

## Development Notes

### Adding New Articles

The pipeline is designed to be rerunnable:
1. Add new articles to `data/articles_restructured.json`
2. Run Stage 1 inference (only processes new articles if implemented)
3. Run Stages 2-3 on new candidates

### Model Files

BERT models are stored in `models/` (not in git). Current model:
- `models/balanced_high_recall_iter0/` - 97.7% recall, 76.8% precision

### DSPy Integration

Stages 2-3 use DSPy for LLM orchestration:
- Signatures define input/output structure
- Supports multiple backends (Claude, Ollama, etc.)
- Metrics enable optimization experiments

### Shared Utilities

The `shared/` module provides common functionality:

```python
from shared import (
    # Configuration
    PROJECT_ROOT, MODEL_CONFIG, STAGE2_CONFIG, STAGE3_CONFIG,

    # Article ID handling (ensures traceability across stages)
    ensure_article_id,        # Add article_id if missing
    normalize_article_fields, # Standardize field names

    # Checkpoint/resume (for long-running processing)
    CheckpointManager,        # Save/load processing state
    filter_unprocessed,       # Skip already-processed articles

    # DSPy helpers
    configure_dspy_lm,        # Configure LLM with retries
    load_optimized_model,     # Load saved DSPy models

    # Logging
    setup_logger,             # Standardized logging setup
    log_section,              # Section headers in logs

    # Validation
    validate_articles,        # Pydantic schema validation
    ArticleBase,              # Base article schema
)
```

### Article ID Consistency

All articles must have a unique `article_id` for traceability:
- If `article_id` exists, it's preserved
- If only `id` exists, it's used as `article_id`
- Otherwise, a deterministic hash ID is generated from content

Use `normalize_article_fields(article)` to ensure consistency.

### Geocode Caching

`stage3/geocode.py` automatically loads cached geocoding results from `results/geocode_cache.json` to avoid redundant API calls. The cache persists between runs.

### Testing

Tests are in `tests/` and cover:
- **Integration tests**: Data flow between stages, checkpoint functionality
- **Extraction tests**: NER accuracy for locations and dates
- **Edge cases**: Empty articles, missing fields, non-Ontario floods

Run tests before making significant changes:
```bash
pytest tests/ -v
```
