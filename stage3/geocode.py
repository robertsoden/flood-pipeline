"""
Stage 3 Geocoding: Convert location names to lat/lon using Mapbox API

Geocodes verified flood locations from stage3_verified.json
"""
import sys
from pathlib import Path
import json
import requests
import time
import argparse
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
from dotenv import load_dotenv
import os
load_dotenv(PROJECT_ROOT / '.env')

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

parser = argparse.ArgumentParser(description='Geocode flood locations using Mapbox API')
parser.add_argument('--api-key', type=str, default=None,
                    help='Mapbox API key (or set MAPBOX_API_KEY env var)')
parser.add_argument('--input', type=str, default='stage3_verified.json',
                    help='Input file (default: stage3_verified.json)')
parser.add_argument('--sample', type=int, default=None,
                    help='Process only first N articles (for testing)')
parser.add_argument('--delay', type=float, default=0.1,
                    help='Delay between API calls in seconds (default: 0.1)')
args = parser.parse_args()

# Get API key
MAPBOX_API_KEY = args.api_key or os.environ.get('MAPBOX_API_KEY')
if not MAPBOX_API_KEY:
    print("ERROR: Mapbox API key required. Either:")
    print("  1. Set MAPBOX_API_KEY environment variable")
    print("  2. Pass --api-key argument")
    sys.exit(1)

print("\n" + "="*70)
print("STAGE 3: GEOCODING FLOOD LOCATIONS")
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================

INPUT_FILE = PROJECT_ROOT / 'results' / args.input
OUTPUT_DIR = PROJECT_ROOT / 'results'

print(f"\nConfiguration:")
print(f"  Input: {INPUT_FILE}")
print(f"  API delay: {args.delay}s between calls")
if args.sample:
    print(f"  Sample size: {args.sample}")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n1. Loading verified articles...")

with open(INPUT_FILE, 'r') as f:
    articles = json.load(f)

print(f"   Loaded {len(articles):,} articles")

if args.sample:
    articles = articles[:args.sample]
    print(f"   Processing sample of {len(articles)} articles")

# ============================================================================
# GEOCODING FUNCTIONS
# ============================================================================

# Cache to avoid duplicate API calls
geocode_cache = {}

# Load existing cache if available
CACHE_FILE = OUTPUT_DIR / 'geocode_cache.json'
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, 'r') as f:
            geocode_cache = json.load(f)
        print(f"   Loaded {len(geocode_cache)} cached locations from {CACHE_FILE}")
    except Exception as e:
        print(f"   Warning: Could not load cache file: {e}")
        geocode_cache = {}

# Ontario bounding box (approximate)
ONTARIO_BOUNDS = {
    'min_lat': 41.7,   # Southern tip (Pelee Island)
    'max_lat': 56.9,   # Northern boundary
    'min_lon': -95.2,  # Western boundary (near Manitoba)
    'max_lon': -74.3   # Eastern boundary (near Quebec)
}

# Known Ontario locations with ambiguous names (fallback coordinates)
# These are locations that commonly geocode incorrectly
KNOWN_ONTARIO_LOCATIONS = {
    # Toronto and GTA general terms
    'greater toronto area': {'lat': 43.70, 'lon': -79.42, 'place_name': 'Greater Toronto Area, Ontario'},
    'toronto area': {'lat': 43.70, 'lon': -79.42, 'place_name': 'Toronto Area, Ontario'},
    'gta': {'lat': 43.70, 'lon': -79.42, 'place_name': 'Greater Toronto Area, Ontario'},

    # Toronto-area rivers and creeks
    'humber river': {'lat': 43.65, 'lon': -79.49, 'place_name': 'Humber River, Toronto, Ontario'},
    'don river': {'lat': 43.70, 'lon': -79.35, 'place_name': 'Don River, Toronto, Ontario'},
    'rouge river': {'lat': 43.80, 'lon': -79.15, 'place_name': 'Rouge River, Toronto, Ontario'},
    'credit river': {'lat': 43.55, 'lon': -79.60, 'place_name': 'Credit River, Mississauga, Ontario'},
    'etobicoke creek': {'lat': 43.65, 'lon': -79.60, 'place_name': 'Etobicoke Creek, Ontario'},
    'highland creek': {'lat': 43.78, 'lon': -79.19, 'place_name': 'Highland Creek, Scarborough, Ontario'},
    'mimico creek': {'lat': 43.65, 'lon': -79.53, 'place_name': 'Mimico Creek, Toronto, Ontario'},
    'black creek': {'lat': 43.72, 'lon': -79.49, 'place_name': 'Black Creek, Toronto, Ontario'},

    # Holland Marsh area
    'holland marsh': {'lat': 44.10, 'lon': -79.55, 'place_name': 'Holland Marsh, Ontario'},
    'holland landing': {'lat': 44.10, 'lon': -79.49, 'place_name': 'Holland Landing, Ontario'},
    'holland river': {'lat': 44.10, 'lon': -79.50, 'place_name': 'Holland River, Ontario'},

    # Grand River watershed
    'grand river': {'lat': 43.40, 'lon': -80.40, 'place_name': 'Grand River, Ontario'},
    'orangeville': {'lat': 43.92, 'lon': -80.09, 'place_name': 'Orangeville, Ontario'},
    'waldemar': {'lat': 43.93, 'lon': -80.15, 'place_name': 'Waldemar, Ontario'},
    'six nations': {'lat': 43.06, 'lon': -80.11, 'place_name': 'Six Nations, Ontario'},

    # Thames River watershed
    'thames river': {'lat': 42.98, 'lon': -81.25, 'place_name': 'Thames River, London, Ontario'},
    'chatham-kent': {'lat': 42.41, 'lon': -82.19, 'place_name': 'Chatham-Kent, Ontario'},

    # Eastern Ontario rivers
    'rideau river': {'lat': 45.42, 'lon': -75.70, 'place_name': 'Rideau River, Ottawa, Ontario'},
    'moira river': {'lat': 44.17, 'lon': -77.38, 'place_name': 'Moira River, Belleville, Ontario'},
    'trent river': {'lat': 44.10, 'lon': -77.58, 'place_name': 'Trent River, Ontario'},
    'otonabee river': {'lat': 44.30, 'lon': -78.32, 'place_name': 'Otonabee River, Peterborough, Ontario'},
    'madawaska river': {'lat': 45.43, 'lon': -77.10, 'place_name': 'Madawaska River, Ontario'},
    # Note: Ontario's Mississippi River (not the US one)
    'mississippi river': {'lat': 45.15, 'lon': -76.25, 'place_name': 'Mississippi River, Lanark County, Ontario'},

    # Central/Northern Ontario
    'muskoka river': {'lat': 45.03, 'lon': -79.30, 'place_name': 'Muskoka River, Ontario'},
    'spanish river': {'lat': 46.20, 'lon': -81.90, 'place_name': 'Spanish River, Ontario'},
    'magnetawan river': {'lat': 45.67, 'lon': -79.63, 'place_name': 'Magnetawan River, Ontario'},
    'french river': {'lat': 46.04, 'lon': -80.77, 'place_name': 'French River, Ontario'},
    'mattagami river': {'lat': 49.70, 'lon': -82.50, 'place_name': 'Mattagami River, Ontario'},

    # Lake Huron / Georgian Bay watersheds
    'saugeen river': {'lat': 44.50, 'lon': -81.37, 'place_name': 'Saugeen River, Southampton, Ontario'},
    'nottawasaga river': {'lat': 44.33, 'lon': -79.93, 'place_name': 'Nottawasaga River, Ontario'},
    'severn river': {'lat': 44.80, 'lon': -79.72, 'place_name': 'Severn River, Ontario'},
}


def is_in_ontario_bounds(lat: float, lon: float) -> bool:
    """Check if coordinates are within Ontario's bounding box."""
    return (ONTARIO_BOUNDS['min_lat'] <= lat <= ONTARIO_BOUNDS['max_lat'] and
            ONTARIO_BOUNDS['min_lon'] <= lon <= ONTARIO_BOUNDS['max_lon'])


def check_ontario_in_context(feature: dict) -> bool:
    """
    Robustly check if a Mapbox feature is in Ontario.
    Checks the context array for region (province) level.
    """
    context = feature.get('context', [])

    # Look for region (province) in context
    for ctx in context:
        ctx_id = ctx.get('id', '')
        ctx_text = ctx.get('text', '')
        # Region ID for Ontario typically contains 'region' and the text is 'Ontario'
        if 'region' in ctx_id and ctx_text == 'Ontario':
            return True

    # Also check place_name for ', Ontario,' pattern (not just 'Ontario' anywhere)
    place_name = feature.get('place_name', '')
    if ', Ontario,' in place_name or place_name.endswith(', Ontario'):
        return True

    return False


def geocode_location(location: str) -> dict:
    """
    Geocode a location string using Mapbox API.
    Returns dict with lat, lon, place_name, and confidence.

    Uses known location fallbacks for ambiguous Ontario place names.
    """
    if not location or location.lower() == 'not found':
        return None

    # Check cache
    cache_key = location.lower().strip()
    if cache_key in geocode_cache:
        return geocode_cache[cache_key]

    # Check for known Ontario locations first (handles ambiguous names)
    location_lower = location.lower()
    for known_name, known_coords in KNOWN_ONTARIO_LOCATIONS.items():
        if known_name in location_lower:
            # Use known coordinates for ambiguous locations
            result = {
                'lat': known_coords['lat'],
                'lon': known_coords['lon'],
                'place_name': known_coords['place_name'],
                'relevance': 0.95,  # High confidence for known locations
                'in_ontario': True,
                'source': 'known_location'
            }
            geocode_cache[cache_key] = result
            return result

    # Add Ontario, Canada context for better results
    query = f"{location}, Ontario, Canada"

    # Mapbox Geocoding API
    url = "https://api.mapbox.com/geocoding/v5/mapbox.places/{}.json".format(
        requests.utils.quote(query)
    )

    params = {
        'access_token': MAPBOX_API_KEY,
        'country': 'CA',  # Limit to Canada
        'limit': 1,
        'types': 'place,locality,neighborhood,address,poi'
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get('features') and len(data['features']) > 0:
            feature = data['features'][0]
            lat = feature['center'][1]
            lon = feature['center'][0]

            is_ontario_context = check_ontario_in_context(feature)
            is_ontario_bounds = is_in_ontario_bounds(lat, lon)
            in_ontario = is_ontario_context and is_ontario_bounds

            result = {
                'lat': lat,
                'lon': lon,
                'place_name': feature.get('place_name', ''),
                'relevance': feature.get('relevance', 0),
                'place_type': feature.get('place_type', []),
                'in_ontario': in_ontario,
                'source': 'mapbox'
            }

            geocode_cache[cache_key] = result
            return result
        else:
            geocode_cache[cache_key] = None
            return None

    except Exception as e:
        print(f"   Warning: Geocoding error for '{location}': {e}")
        return None


# ============================================================================
# GEOCODE LOCATIONS
# ============================================================================

print("\n2. Geocoding locations...")

stats = {
    'total': 0,
    'geocoded': 0,
    'in_ontario': 0,
    'not_found': 0,
    'no_location': 0,
    'cached': 0,
}

# Get unique locations first to minimize API calls
unique_locations = set()
for art in articles:
    s3 = art.get('stage3', {})
    loc = s3.get('location', '')
    if loc and loc.lower() != 'not found':
        unique_locations.add(loc)

print(f"   Found {len(unique_locations)} unique locations to geocode")

# Geocode unique locations with progress bar
print("\n   Geocoding unique locations...")
for loc in tqdm(unique_locations, desc="   Geocoding"):
    if loc.lower() not in geocode_cache:
        geocode_location(loc)
        time.sleep(args.delay)  # Rate limiting

print(f"   Cache size: {len(geocode_cache)} locations")

# Apply geocoding results to all articles
print("\n   Applying results to articles...")
for art in tqdm(articles, desc="   Applying"):
    s3 = art.get('stage3', {})
    loc = s3.get('location', '')
    stats['total'] += 1

    if not loc or loc.lower() == 'not found':
        stats['no_location'] += 1
        continue

    result = geocode_cache.get(loc.lower().strip())

    if result:
        s3['geocode'] = {
            'lat': result['lat'],
            'lon': result['lon'],
            'place_name': result['place_name'],
            'relevance': result['relevance'],
            'in_ontario': result['in_ontario'],
        }
        stats['geocoded'] += 1
        if result['in_ontario']:
            stats['in_ontario'] += 1
    else:
        stats['not_found'] += 1

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n3. Saving results...")

# Save geocoded articles
if args.sample:
    output_file = OUTPUT_DIR / f'stage3_geocoded_sample_{args.sample}.json'
else:
    output_file = OUTPUT_DIR / 'stage3_geocoded.json'

with open(output_file, 'w') as f:
    json.dump(articles, f, indent=2)
print(f"   Saved: {output_file}")

# Save geocode cache for reuse
cache_file = OUTPUT_DIR / 'geocode_cache.json'
with open(cache_file, 'w') as f:
    json.dump(geocode_cache, f, indent=2)
print(f"   Saved cache: {cache_file}")

# Save summary
summary = {
    'input_file': str(INPUT_FILE),
    'total_articles': stats['total'],
    'unique_locations': len(unique_locations),
    'statistics': stats,
    'geocode_rate': stats['geocoded'] / max(stats['total'], 1),
    'ontario_rate': stats['in_ontario'] / max(stats['geocoded'], 1),
}

if args.sample:
    summary_file = OUTPUT_DIR / f'stage3_geocoded_sample_{args.sample}_summary.json'
else:
    summary_file = OUTPUT_DIR / 'stage3_geocoded_summary.json'

with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   Saved summary: {summary_file}")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("GEOCODING COMPLETE")
print("="*70)

print(f"\nResults:")
print(f"  Total articles:     {stats['total']:,}")
print(f"  No location:        {stats['no_location']:,}")
print(f"  Geocoded:           {stats['geocoded']:,} ({stats['geocoded']/max(stats['total']-stats['no_location'],1):.1%})")
print(f"  In Ontario:         {stats['in_ontario']:,} ({stats['in_ontario']/max(stats['geocoded'],1):.1%})")
print(f"  Not found:          {stats['not_found']:,}")

print(f"\nOutput files:")
print(f"  {output_file}")
print(f"  {summary_file}")
print(f"  {cache_file}")

# Show sample results
print("\n" + "="*70)
print("SAMPLE GEOCODED RESULTS")
print("="*70)

count = 0
for art in articles:
    if count >= 5:
        break
    s3 = art.get('stage3', {})
    if 'geocode' in s3:
        count += 1
        geo = s3['geocode']
        print(f"\n{count}. {s3.get('location', 'N/A')}")
        print(f"   -> {geo['place_name']}")
        print(f"   -> lat: {geo['lat']:.4f}, lon: {geo['lon']:.4f}")
        print(f"   -> In Ontario: {geo['in_ontario']}, Relevance: {geo['relevance']:.2f}")

print("\n" + "="*70 + "\n")
