#!/usr/bin/env python3
"""
Download Ontario historical newspapers from Internet Archive.

All collections with bulk download access:
- CABHC (Belleville/Hastings): 5,000+ items, 1834-present
- Haliburton Highlands: 7,000+ items
- Peterborough Examiner: 66 items, 1935-1958
- Napanee Express: 1,466 items, 1868-1918
- Napanee Beaver: 258 items, 1884-1897
- Grimsby Independent: 50 items, 1885-1890
- LaSalle News: 92 items, 1953-1962
- Fort Erie Times: 371 items, 2009-2010
- At Guelph: 68 items, 1987-1996
- Inport News: 384 items, 2010-2016

Usage:
    python scripts/download_ia_newspapers.py [--collection NAME] [--dry-run]

Requires:
    pip install internetarchive
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

try:
    import internetarchive as ia
except ImportError:
    print("ERROR: internetarchive package required")
    print("Install with: pip install internetarchive")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "ia_newspapers"

# All Ontario newspaper collections
COLLECTIONS = {
    # Community Archives - largest historical collection
    'cabhc': {
        'query': 'collection:cabhc_newspapers',
        'name': 'Community Archives of Belleville and Hastings County',
        'region': 'Belleville, Hastings County (SE Ontario)',
        'years': '1834-present',
    },
    # Haliburton Highlands
    'haliburton': {
        'query': 'collection:hhda-newspapers',
        'name': 'Haliburton Highlands Digital Archives',
        'region': 'Haliburton County (North-Central Ontario)',
        'years': 'Various',
    },
    # Peterborough - use title search since no dedicated collection
    'peterborough': {
        'query': 'title:"peterborough examiner" AND mediatype:texts',
        'name': 'Peterborough Examiner',
        'region': 'Peterborough County (Central Ontario)',
        'years': '1935-1958',
    },
    # Napanee newspapers - excellent historical coverage
    'napanee_express': {
        'query': 'collection:napanee-express',
        'name': 'Napanee Express',
        'region': 'Napanee, Ontario',
        'years': '1868-1918',
    },
    'napanee_beaver': {
        'query': 'collection:napanee-beaver-collection',
        'name': 'Napanee Beaver',
        'region': 'Napanee, Ontario',
        'years': '1884-1897',
    },
    # Other Ontario newspapers
    'grimsby': {
        'query': 'collection:grimsby-independent',
        'name': 'Grimsby Independent',
        'region': 'Grimsby, Ontario',
        'years': '1885-1890',
    },
    'lasalle': {
        'query': 'collection:lasalle-news-lasalle-ontario',
        'name': 'LaSalle News',
        'region': 'LaSalle, Ontario',
        'years': '1953-1962',
    },
    'fort_erie': {
        'query': 'collection:fort-erie-times',
        'name': 'Fort Erie Times',
        'region': 'Fort Erie, Ontario',
        'years': '2009-2010',
    },
    'at_guelph': {
        'query': 'collection:atguelph',
        'name': 'At Guelph',
        'region': 'Guelph, Ontario',
        'years': '1987-1996',
    },
    'inport': {
        'query': 'collection:inport-news',
        'name': 'Inport News',
        'region': 'Port Colborne, Ontario',
        'years': '2010-2016',
    },
}


def download_collection(key, output_dir, dry_run=False, limit=None):
    """Download all items from a collection."""

    if key not in COLLECTIONS:
        print(f"ERROR: Unknown collection '{key}'")
        print(f"Available: {', '.join(COLLECTIONS.keys())}")
        return {}

    info = COLLECTIONS[key]
    collection_dir = output_dir / key

    print(f"\n{'='*70}")
    print(f"COLLECTION: {info['name']}")
    print(f"{'='*70}")
    print(f"Region: {info['region']}")
    print(f"Years: {info['years']}")
    print(f"Query: {info['query']}")
    print()

    # Search for items
    items = list(ia.search_items(info['query']))
    print(f"Found {len(items)} items")

    if limit:
        items = items[:limit]
        print(f"Limited to {limit} items")

    if dry_run:
        print("\nDRY RUN - Items that would be downloaded:")
        for i, item in enumerate(items[:20]):
            print(f"  [{i+1}] {item['identifier']}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more")
        return {'items': len(items), 'downloaded': 0}

    collection_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, item_info in enumerate(items):
        identifier = item_info['identifier']
        print(f"[{i+1}/{len(items)}] {identifier}...", end=' ', flush=True)

        item_dir = collection_dir / identifier

        try:
            item = ia.get_item(identifier)
            metadata = {
                'title': item.metadata.get('title'),
                'date': item.metadata.get('date'),
                'description': item.metadata.get('description'),
            }

            # Download text files
            files_downloaded = []
            for pattern in ['*_djvu.txt', '*.txt']:
                for f in item.get_files(glob_pattern=pattern):
                    if pattern == '*.txt' and '_djvu.txt' in f.name:
                        continue
                    dest_path = item_dir / f.name
                    if not dest_path.exists():
                        item_dir.mkdir(parents=True, exist_ok=True)
                        f.download(destdir=str(item_dir))
                        files_downloaded.append(f.name)

            results.append({
                'identifier': identifier,
                'metadata': metadata,
                'files_downloaded': files_downloaded,
                'error': None
            })

            if files_downloaded:
                print(f"Downloaded {len(files_downloaded)} files")
            else:
                print("Skipped (exists or no text)")

        except Exception as e:
            results.append({
                'identifier': identifier,
                'error': str(e)
            })
            print(f"ERROR: {e}")

    # Save manifest
    manifest = {
        'collection': info,
        'downloaded_at': datetime.now().isoformat(),
        'total_items': len(items),
        'results': results,
    }

    with open(collection_dir / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    downloaded = len([r for r in results if r.get('files_downloaded')])
    print(f"\nCompleted: {downloaded}/{len(items)} items with text files")
    return {'items': len(items), 'downloaded': downloaded}


def main():
    parser = argparse.ArgumentParser(
        description='Download Ontario newspapers from Internet Archive'
    )
    parser.add_argument(
        '--collection', '-c',
        choices=list(COLLECTIONS.keys()) + ['all'],
        default='all',
        help='Collection to download'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT,
        help='Output directory'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='List items without downloading'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        help='Limit items per collection (for testing)'
    )

    args = parser.parse_args()

    print("="*70)
    print("INTERNET ARCHIVE ONTARIO NEWSPAPERS DOWNLOADER")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output: {args.output}")
    print(f"Dry run: {args.dry_run}")

    args.output.mkdir(parents=True, exist_ok=True)

    collections = list(COLLECTIONS.keys()) if args.collection == 'all' else [args.collection]

    total_items = 0
    total_downloaded = 0

    for key in collections:
        result = download_collection(key, args.output, args.dry_run, args.limit)
        total_items += result.get('items', 0)
        total_downloaded += result.get('downloaded', 0)

    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"Total items: {total_items}")
    print(f"Total downloaded: {total_downloaded}")
    print(f"\nNext step: Prepare for BERT")
    print(f"  python scripts/prepare_ia_for_bert.py")


if __name__ == '__main__':
    main()
