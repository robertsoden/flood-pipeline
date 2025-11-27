# Ontario Flood Article Extraction Pipeline

NLP pipeline to identify and extract flood information from 50,000+ Ontario newspaper articles.

## Overview

This pipeline extracts **when** and **where** flood events occurred from historical newspaper articles. It produces geocoded flood records that can be integrated with other flood databases for triangulation and analysis.

### Results

| Stage | Description | Output |
|-------|-------------|--------|
| Stage 1 | BERT classification | 5,247 flood candidates |
| Stage 2 | LLM flood verification | 2,290 confirmed Ontario floods |
| Stage 3 | NER + Geocoding | 1,940 geocoded articles |

## Quick Start

```bash
# Setup
python -m venv env
source env/bin/activate
pip install -r requirements.txt

# Configure API keys
cp env.example .env
# Edit .env with MAPBOX_TOKEN and ANTHROPIC_API_KEY

# Run pipeline
python stage1-bert/bert-inference.py       # BERT filter
python stage2/process.py                   # LLM verification
python stage3/process_ner.py               # NER extraction
python stage3/process_llm_verify.py        # LLM date/location verification
python stage3/geocode.py                   # Geocode to lat/lon
```

## Pipeline Architecture

```
Raw Articles (50K) → BERT Filter → LLM Verify → NER + Geocode → Geocoded Floods
                     (Stage 1)     (Stage 2)      (Stage 3)
```

**Stage 1 (BERT):** High-recall filter (97.7%) using semi-supervised learning. Removes clearly non-flood articles.

**Stage 2 (LLM):** Verifies actual flood events (not just mentions) and confirms Ontario location using Claude.

**Stage 3 (NER + Geocoding):** Extracts flood dates and locations using spaCy NER, verifies/corrects with LLM, geocodes to coordinates via Mapbox.

## Output Format

Final output in `results/stage3_geocoded.json`:

```json
{
  "article_id": "12345",
  "full_text": "Flooding hit Toronto...",
  "publication_date": "2013-07-08",
  "stage3": {
    "location": "Toronto",
    "flood_date": "July 2013",
    "latitude": 43.6532,
    "longitude": -79.3832,
    "date_confidence": "high"
  }
}
```

## Project Structure

```
flood_pipeline/
├── shared/           # Config and utilities
├── stage1-bert/      # BERT classification
├── stage2/           # LLM flood verification
├── stage3/           # NER + geocoding
├── results/          # Pipeline outputs (not in git)
├── models/           # Trained models (not in git)
└── data/             # Raw articles (not in git)
```

## Requirements

- Python 3.10+
- BERT model for Stage 1 (trained locally)
- Anthropic API key for Claude (Stages 2-3)
- Mapbox API token for geocoding

## Integration

Output feeds into the [flood_mcp](../flood_mcp) triangulation system where articles are combined with database records (CDD, NRCAN, OEM, Conservation Ontario) to build a comprehensive flood history.

## Documentation

- `CLAUDE.md` - Detailed technical documentation for AI assistants
- `PIPELINE_STATUS.md` - Current processing status and metrics
