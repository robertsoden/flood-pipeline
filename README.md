# Flood History of Ontario

Multi-stage pipeline to identify and extract flood information from 50,000+ Ontario newspaper articles.

## Pipeline Overview

**Stage 1 (BERT):** High-recall filter (>95%) to exclude non-flood articles  
**Stage 2 (Local LLM):** High-precision flood verification + Ontario location check  
**Stage 3 (Local LLM + Geocoding):** Extract flood date/location, geocode to lat/lon  
**Stage 4 (External LLM):** Extract detailed flood impacts  

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure (copy and edit)
cp .env.example .env

# Access Data
This assumes access to our archive of newspaper articles which is not stored in github

# Run pipeline
python stage1-bert/bert-train.py      # Train BERT (or use pre-trained)
python stage1-bert/bert-inference.py  # Filter 50K articles
python stage2/verify.py               # Verify floods (to be implemented)
python stage3/extract.py              # Extract location/date (to be implemented)
python stage4/impacts.py              # Extract impacts (to be implemented)
```

## Project Structure

```
shared/              # Shared config and utilities
  ├── config.py      # All configuration (paths, models, settings)
  └── utils.py       # Shared functions (JSON I/O, logging, caching)

stage1-bert/         # BERT-based filtering (✅ Complete)
  ├── bert-train.py
  ├── bert-inference.py
  └── data/

stage2/              # LLM flood verification (🔄 In progress)
stage3/              # Location/date extraction (🔄 To do)
stage4/              # Impact extraction (🔄 To do)

annotations/         # Manual labeling tool
data/                # Raw articles (not in git)
models/              # Trained models (not in git)
results/             # Pipeline outputs (not in git)
```

## Data Flow

Articles are progressively enriched through each stage:

```
Raw articles (50K) 
  → Stage 1: Filter to ~5-15K candidates
  → Stage 2: Verify ~2-8K confirmed Ontario floods  
  → Stage 3: Geolocate floods with dates
  → Stage 4: Extract detailed impacts
```

Each stage adds a `stageN` key to article JSON with its results.

## Configuration

All settings in `shared/config.py`:
- File paths
- Model configurations (Ollama, BERT, external APIs)
- Stage-specific parameters
- Geocoding settings

See `.env.example` for required environment variables (API keys, etc.)


## Documentation

- `PROJECT_CONTEXT.md` - Comprehensive project documentation
- `VERSION_CONTROL_GUIDE.md` - Git best practices
- `DATA_README.md` - Data sources and download instructions

## Status

- ✅ Stage 1: BERT filtering complete
- 🔄 Stage 2-4: In development
- 📊 Data: 580 labeled examples, 50K unlabeled articles