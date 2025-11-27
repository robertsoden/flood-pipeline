# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Purpose:** Extract flood event information from 50,000+ Ontario newspaper articles using a 3-stage NLP pipeline.

**Output:** Geocoded flood articles with dates and locations, ready for integration with the `flood_mcp` triangulation system.

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
    Export to flood_mcp for triangulation with database sources
```

## Directory Structure

```
flood_pipeline/
├── shared/                 # Shared configuration and utilities
│   ├── config.py          # Central configuration (paths, models, settings)
│   └── utils.py           # JSON I/O, data prep utilities
│
├── stage1-bert/           # BERT-based filtering (✅ Complete)
│   ├── bert-train.py      # Semi-supervised training with pseudo-labeling
│   ├── bert-inference.py  # Apply trained model to full dataset
│   └── data/              # Training/test datasets
│
├── stage2/                # LLM flood verification (✅ Complete)
│   ├── process.py         # Main processing script
│   ├── signatures.py      # DSPy signatures for verification
│   └── metrics.py         # Evaluation metrics
│
├── stage3/                # Location/date extraction (✅ Complete)
│   ├── process_ner.py     # NER extraction with spaCy
│   ├── process_llm_verify.py  # LLM verification of extracted data
│   ├── geocode.py         # Mapbox geocoding
│   └── signatures.py      # DSPy signatures for extraction
│
├── results/               # Pipeline outputs (not in git)
│   ├── stage2_ontario_floods.json
│   ├── stage3_verified.json
│   └── stage3_geocoded.json  # Final output
│
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

### Environment Setup

```bash
# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

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

## Integration with flood_mcp

This pipeline's output (`results/stage3_geocoded.json`) feeds into the `flood_mcp` triangulation system where:

1. Articles become **cases** in the unified schema
2. Cases cluster into **events** with database records
3. Events receive confidence scores from multiple sources
4. Feedback loop enables verification/rejection

The pipeline focuses on NLP extraction; impact data and triangulation happen downstream.

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
