#!/usr/bin/env python3
"""
Convert Internet Archive newspaper text to format expected by flood_news BERT pipeline.

This script:
1. Reads downloaded IA DJVU text files
2. Splits into article-sized chunks
3. Outputs JSON in flood_news format for BERT classification

Usage:
    python scripts/prepare_ia_for_bert.py

Output:
    data/ia_articles.json  (ready for BERT inference)
"""

import json
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
IA_DIR = PROJECT_ROOT / "data" / "ia_newspapers"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ia_articles.json"

# Chunk settings - split long texts into article-sized pieces
MAX_CHUNK_SIZE = 4000  # characters per chunk (BERT max is ~512 tokens ≈ 2000 chars, but we want context)
MIN_CHUNK_SIZE = 200   # skip very short chunks
OVERLAP = 200          # overlap between chunks to avoid splitting articles


def load_manifest(collection_dir: Path) -> dict:
    """Load manifest for metadata."""
    manifest_file = collection_dir / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        return {r['identifier']: r.get('metadata', {}) for r in manifest.get('results', [])}
    return {}


def extract_date_from_identifier(identifier: str, metadata: dict = None) -> str:
    """Extract date string from identifier or metadata."""

    # Try metadata first
    if metadata and metadata.get('date'):
        return metadata['date']

    # ISO date: YYYY-MM-DD
    match = re.search(r'(\d{4}-\d{2}-\d{2})', identifier)
    if match:
        return match.group(1)

    # Month + year: january-1924
    month_names = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
    }
    for month, num in month_names.items():
        match = re.search(rf'{month}[-_]?(\d{{4}})', identifier.lower())
        if match:
            return f"{match.group(1)}-{num}"

    # Just year
    match = re.search(r'(\d{4})', identifier)
    if match:
        year = int(match.group(1))
        if 1800 <= year <= 2025:
            return str(year)

    return ""


def extract_location_from_metadata(metadata: dict) -> str:
    """Extract location from metadata description."""
    if not metadata:
        return ""

    desc = metadata.get('description', '') or ''
    title = metadata.get('title', '') or ''

    # Look for "published in X, Ontario"
    match = re.search(r'published in ([^,]+),?\s*(?:Ontario|Ont)', desc, re.I)
    if match:
        return f"{match.group(1)}, Ontario"

    # Common newspaper locations
    locations = {
        'belleville': 'Belleville, Ontario',
        'peterborough': 'Peterborough, Ontario',
        'hastings': 'Hastings County, Ontario',
        'trenton': 'Trenton, Ontario',
        'haliburton': 'Haliburton, Ontario',
        'deseronto': 'Deseronto, Ontario',
        'madoc': 'Madoc, Ontario',
        'stirling': 'Stirling, Ontario',
        'napanee': 'Napanee, Ontario',
        'grimsby': 'Grimsby, Ontario',
        'lasalle': 'LaSalle, Ontario',
        'fort erie': 'Fort Erie, Ontario',
        'guelph': 'Guelph, Ontario',
        'port colborne': 'Port Colborne, Ontario',
    }

    text = (desc + ' ' + title).lower()
    for key, loc in locations.items():
        if key in text:
            return loc

    return "Ontario"


def split_into_chunks(text: str, identifier: str) -> list:
    """Split long text into article-sized chunks."""

    if len(text) <= MAX_CHUNK_SIZE:
        return [text] if len(text) >= MIN_CHUNK_SIZE else []

    chunks = []

    # Try to split on paragraph breaks
    paragraphs = re.split(r'\n\s*\n', text)

    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) <= MAX_CHUNK_SIZE:
            current_chunk += para + "\n\n"
        else:
            if len(current_chunk) >= MIN_CHUNK_SIZE:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"

    # Don't forget last chunk
    if len(current_chunk) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())

    # If paragraph splitting didn't work well, fall back to character splitting
    if not chunks:
        for i in range(0, len(text), MAX_CHUNK_SIZE - OVERLAP):
            chunk = text[i:i + MAX_CHUNK_SIZE]
            if len(chunk) >= MIN_CHUNK_SIZE:
                chunks.append(chunk)

    return chunks


def process_text_file(file_path: Path, identifier: str, metadata: dict) -> list:
    """Process a single text file into article records."""

    try:
        try:
            text = file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            text = file_path.read_text(encoding='latin-1')
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return []

    # Clean text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if len(text) < MIN_CHUNK_SIZE:
        return []

    # Split into chunks
    chunks = split_into_chunks(text, identifier)

    # Extract metadata
    date = extract_date_from_identifier(identifier, metadata)
    location = extract_location_from_metadata(metadata)
    title = metadata.get('title', '') if metadata else ''

    # Create article records
    articles = []
    for i, chunk in enumerate(chunks):
        article_id = f"ia-{identifier}-{i}" if len(chunks) > 1 else f"ia-{identifier}"

        articles.append({
            'id': article_id,
            'article_id': article_id,
            'title': title or f"Article from {identifier}",
            'full_text': chunk,
            'date': date,
            'location': location,
            'publisher': metadata.get('title', '') if metadata else identifier,
            'source': 'internet_archive',
            'ia_identifier': identifier,
            'ia_collection': file_path.parent.parent.name,
        })

    return articles


def main():
    print("=" * 70)
    print("PREPARE INTERNET ARCHIVE CONTENT FOR BERT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Input: {IA_DIR}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    if not IA_DIR.exists():
        print(f"ERROR: IA directory not found: {IA_DIR}")
        print("Run the download script first:")
        print("  python scripts/download_ia_newspapers.py")
        return

    all_articles = []

    # Find all collection directories
    collections = [d for d in IA_DIR.iterdir() if d.is_dir()]
    print(f"Found {len(collections)} collections")

    for collection_dir in collections:
        collection = collection_dir.name
        print(f"\nProcessing: {collection}")

        # Load manifest
        metadata_lookup = load_manifest(collection_dir)

        # Find item directories
        items = [d for d in collection_dir.iterdir() if d.is_dir()]
        print(f"  {len(items)} items")

        collection_articles = []
        for item_dir in items:
            identifier = item_dir.name
            metadata = metadata_lookup.get(identifier, {})

            # Find text files
            text_files = list(item_dir.glob('*_djvu.txt')) + list(item_dir.glob('*.txt'))

            for text_file in text_files:
                articles = process_text_file(text_file, identifier, metadata)
                collection_articles.extend(articles)

        print(f"  → {len(collection_articles)} article chunks")
        all_articles.extend(collection_articles)

    print(f"\n{'='*70}")
    print(f"TOTAL: {len(all_articles)} article chunks")
    print(f"{'='*70}")

    # Save output
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_articles, f, indent=2)

    print(f"\nSaved to: {OUTPUT_FILE}")
    print(f"\nNext step: Run BERT inference")
    print(f"  python stage1-bert/bert-inference.py --input data/ia_articles.json")


if __name__ == '__main__':
    main()
