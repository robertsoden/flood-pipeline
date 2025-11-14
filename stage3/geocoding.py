"""
Stage 3: Geocoding Module
Geocode extracted locations using Mapbox (preferred) or Nominatim (fallback)
"""
import sys
from pathlib import Path
import time
import requests
from typing import Optional, Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils import load_json, save_json, setup_logging


class GeocodingCache:
    """Cache for geocoding results to avoid redundant API calls"""
    
    def __init__(self, cache_file: Path):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.hits = 0
        self.misses = 0
    
    def _load_cache(self) -> dict:
        """Load cache from file"""
        if self.cache_file.exists():
            return load_json(self.cache_file)
        return {}
    
    def _save_cache(self):
        """Save cache to file"""
        save_json(self.cache, self.cache_file)
    
    def get(self, location: str) -> Optional[Dict]:
        """Get cached geocoding result"""
        key = location.lower().strip()
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None
    
    def set(self, location: str, lat: float, lon: float, source: str = 'unknown'):
        """Cache geocoding result"""
        key = location.lower().strip()
        self.cache[key] = {
            'lat': lat,
            'lon': lon,
            'source': source
        }
        self._save_cache()
    
    def stats(self) -> str:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache: {len(self.cache)} entries, {self.hits} hits, {self.misses} misses ({hit_rate:.1f}% hit rate)"


class Geocoder:
    """Geocoder with multiple backend support"""
    
    def __init__(
        self,
        cache_file: Path,
        mapbox_token: Optional[str] = None,
        nominatim_delay: float = 1.0,
        focus_region: str = "Ontario, Canada",
        logger = None
    ):
        """
        Initialize geocoder
        
        Args:
            cache_file: Path to cache file
            mapbox_token: Mapbox API token (if using Mapbox)
            nominatim_delay: Delay between Nominatim requests (seconds)
            focus_region: Region to focus searches on
            logger: Logger instance
        """
        self.cache = GeocodingCache(cache_file)
        self.mapbox_token = mapbox_token
        self.nominatim_delay = nominatim_delay
        self.focus_region = focus_region
        self.logger = logger or setup_logging('geocoder')
        
        # Toronto coordinates for proximity bias
        self.toronto_coords = (-79.3832, 43.6532)
        
        # Track API usage
        self.mapbox_calls = 0
        self.nominatim_calls = 0
    
    def geocode(self, location: str, country: str = "CA") -> Optional[Dict]:
        """
        Geocode a location string
        
        Args:
            location: Location string to geocode
            country: Country code (default: CA for Canada)
        
        Returns:
            Dictionary with lat, lon, source, or None if failed
        """
        # Check cache first
        cached = self.cache.get(location)
        if cached:
            return cached
        
        # Try Mapbox first (if token provided)
        if self.mapbox_token:
            result = self._geocode_mapbox(location, country)
            if result:
                self.cache.set(location, result['lat'], result['lon'], 'mapbox')
                return result
        
        # Fallback to Nominatim
        result = self._geocode_nominatim(location, country)
        if result:
            self.cache.set(location, result['lat'], result['lon'], 'nominatim')
            return result
        
        # Failed
        self.logger.warning(f"Failed to geocode: {location}")
        return None
    
    def _geocode_mapbox(self, location: str, country: str) -> Optional[Dict]:
        """Geocode using Mapbox API"""
        try:
            url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{location}.json"
            params = {
                'access_token': self.mapbox_token,
                'country': country,
                'proximity': f"{self.toronto_coords[0]},{self.toronto_coords[1]}",
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.mapbox_calls += 1
            
            if data.get('features'):
                feature = data['features'][0]
                lon, lat = feature['geometry']['coordinates']
                return {
                    'lat': lat,
                    'lon': lon,
                    'source': 'mapbox'
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Mapbox geocoding failed for '{location}': {e}")
            return None
    
    def _geocode_nominatim(self, location: str, country: str) -> Optional[Dict]:
        """Geocode using Nominatim (OpenStreetMap) API"""
        try:
            # Rate limiting
            time.sleep(self.nominatim_delay)
            
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': f"{location}, {self.focus_region}",
                'format': 'json',
                'limit': 1,
                'countrycodes': country.lower()
            }
            headers = {
                'User-Agent': 'FloodHistoryOntario/1.0 (Research Project)'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            self.nominatim_calls += 1
            
            if data:
                result = data[0]
                return {
                    'lat': float(result['lat']),
                    'lon': float(result['lon']),
                    'source': 'nominatim'
                }
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Nominatim geocoding failed for '{location}': {e}")
            return None
    
    def stats(self) -> str:
        """Get geocoding statistics"""
        cache_stats = self.cache.stats()
        api_stats = f"API calls: Mapbox={self.mapbox_calls}, Nominatim={self.nominatim_calls}"
        return f"{cache_stats} | {api_stats}"


def batch_geocode(
    locations: list[str],
    cache_file: Path,
    mapbox_token: Optional[str] = None,
    batch_size: int = 100
) -> Dict[str, Dict]:
    """
    Batch geocode a list of locations
    
    Args:
        locations: List of location strings
        cache_file: Path to cache file
        mapbox_token: Optional Mapbox API token
        batch_size: Report progress every N locations
    
    Returns:
        Dictionary mapping location -> geocoding result
    """
    logger = setup_logging('batch_geocode')
    geocoder = Geocoder(cache_file, mapbox_token, logger=logger)
    
    results = {}
    unique_locations = list(set(locations))
    
    logger.info(f"Geocoding {len(unique_locations)} unique locations...")
    
    for i, location in enumerate(unique_locations, 1):
        result = geocoder.geocode(location)
        results[location] = result
        
        if i % batch_size == 0 or i == len(unique_locations):
            logger.info(f"Progress: {i}/{len(unique_locations)} | {geocoder.stats()}")
    
    # Final stats
    success_count = sum(1 for r in results.values() if r is not None)
    success_rate = success_count / len(results) * 100 if results else 0
    
    logger.info(f"\nGeocoding complete:")
    logger.info(f"  Success: {success_count}/{len(results)} ({success_rate:.1f}%)")
    logger.info(f"  {geocoder.stats()}")
    
    return results


if __name__ == '__main__':
    # Test geocoding
    cache_file = PROJECT_ROOT / 'stage3' / 'geocoding_cache.json'
    
    test_locations = [
        "Timmins, Ontario",
        "Grand River",
        "Toronto, Ontario",
        "Mattagami River",
        "Northern Ontario"
    ]
    
    results = batch_geocode(test_locations, cache_file)
    
    print("\nResults:")
    for loc, result in results.items():
        if result:
            print(f"  {loc}: ({result['lat']:.4f}, {result['lon']:.4f}) [{result['source']}]")
        else:
            print(f"  {loc}: FAILED")
