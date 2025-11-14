"""
Stage 3: Production Inference
Extract locations and dates from Stage 2 confirmed floods
"""
import sys
from pathlib import Path
import dspy
import time
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.config import MODEL_CONFIG, STAGE3_CONFIG, PROJECT_ROOT as PROJ_ROOT
from shared.utils import load_json, save_json, setup_logging
from stage3.signature import LocationDateExtractor, extract_from_article
from stage3.geocoding import Geocoder


def run_extraction(
    input_file: Path,
    output_file: Path,
    model_file: Path,
    geocode: bool = True,
    mapbox_token: str = None,
    batch_save_interval: int = 1000
):
    """
    Run location and date extraction on all articles
    
    Args:
        input_file: Path to Stage 2 output (confirmed floods)
        output_file: Path to save Stage 3 output
        model_file: Path to trained/optimized extractor model
        geocode: Whether to geocode locations
        mapbox_token: Mapbox API token (optional)
        batch_save_interval: Save progress every N articles
    """
    logger = setup_logging('stage3_extraction')
    
    logger.info("=" * 70)
    logger.info("STAGE 3: LOCATION & DATE EXTRACTION")
    logger.info("=" * 70)
    
    # Configure LLM
    logger.info(f"\n[1/7] Configuring LLM: {MODEL_CONFIG['name']}")
    logger.info(f"Temperature: {STAGE3_CONFIG['temperature_inference']} (inference mode)")
    lm = dspy.LM(
        model=MODEL_CONFIG['name'],
        api_base=MODEL_CONFIG['api_base'],
        api_key=MODEL_CONFIG['api_key'],
        temperature=STAGE3_CONFIG['temperature_inference']  # Lower for deterministic inference
    )
    dspy.configure(lm=lm)
    
    # Load model
    logger.info(f"\n[2/7] Loading extraction model from {model_file}")
    if model_file.exists():
        extractor = LocationDateExtractor()
        extractor.load(str(model_file))
        logger.info("Loaded optimized model")
    else:
        logger.warning(f"Model file not found, using baseline")
        extractor = LocationDateExtractor()
    
    # Load input articles
    logger.info(f"\n[3/7] Loading articles from {input_file}")
    articles = load_json(input_file)
    logger.info(f"Loaded {len(articles)} articles to process")
    
    # Initialize geocoder
    geocoder = None
    if geocode:
        logger.info(f"\n[4/7] Initializing geocoder...")
        cache_file = PROJ_ROOT / 'stage3' / 'geocoding_cache.json'
        geocoder = Geocoder(
            cache_file=cache_file,
            mapbox_token=mapbox_token,
            nominatim_delay=1.0,
            focus_region="Ontario, Canada",
            logger=logger
        )
        logger.info("Geocoder ready")
    else:
        logger.info(f"\n[4/7] Geocoding disabled")
    
    # Process articles
    logger.info(f"\n[5/7] Extracting locations and dates...")
    logger.info(f"Saving progress every {batch_save_interval} articles")
    
    start_time = time.time()
    processed = 0
    errors = 0
    
    for i, article in enumerate(articles, 1):
        try:
            # Extract location and date
            article_text = article.get('article_text', article.get('full_text', ''))
            pub_date = article.get('publication_date', 'unknown')
            
            result = extractor(
                article_text=article_text,
                publication_date=pub_date
            )
            
            # Build stage3 output
            stage3_data = {
                'primary_location': {
                    'name': result.primary_location,
                    'confidence': result.location_confidence,
                    'reasoning': result.location_reasoning
                },
                'flood_date': {
                    'date': result.flood_date,
                    'precision': result.date_precision,
                    'confidence': result.date_confidence,
                    'reasoning': result.date_reasoning
                },
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Geocode primary location
            if geocode and geocoder and result.primary_location.lower() != 'unknown':
                geo_result = geocoder.geocode(result.primary_location)
                if geo_result:
                    stage3_data['primary_location']['lat'] = geo_result['lat']
                    stage3_data['primary_location']['lon'] = geo_result['lon']
                    stage3_data['primary_location']['geocode_source'] = geo_result['source']
                else:
                    stage3_data['primary_location']['lat'] = None
                    stage3_data['primary_location']['lon'] = None
                    stage3_data['primary_location']['geocode_source'] = 'failed'
            
            # Add to article
            article['stage3'] = stage3_data
            processed += 1
            
        except Exception as e:
            logger.error(f"Error processing article {i}: {e}")
            article['stage3'] = {
                'error': str(e),
                'processing_timestamp': datetime.now().isoformat()
            }
            errors += 1
        
        # Progress update
        if i % 100 == 0 or i == len(articles):
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(articles) - i) / rate if rate > 0 else 0
            
            logger.info(f"Progress: {i}/{len(articles)} ({i/len(articles)*100:.1f}%) | "
                       f"Rate: {rate:.1f} articles/sec | ETA: {eta/60:.1f} min")
            
            if geocoder:
                logger.info(f"  {geocoder.stats()}")
        
        # Periodic save
        if i % batch_save_interval == 0:
            logger.info(f"Saving checkpoint to {output_file}")
            save_json(articles, output_file)
    
    # Final save
    logger.info(f"\n[6/7] Saving final results to {output_file}")
    save_json(articles, output_file)
    
    # Statistics
    logger.info(f"\n[7/7] Processing complete!")
    
    total_time = time.time() - start_time
    logger.info(f"\nStatistics:")
    logger.info(f"  Total articles: {len(articles)}")
    logger.info(f"  Successfully processed: {processed}")
    logger.info(f"  Errors: {errors}")
    logger.info(f"  Success rate: {processed/len(articles)*100:.1f}%")
    logger.info(f"  Total time: {total_time/60:.1f} minutes")
    logger.info(f"  Average: {total_time/len(articles):.2f} seconds/article")
    
    if geocoder:
        logger.info(f"\nGeocoding:")
        logger.info(f"  {geocoder.stats()}")
        
        # Count successful geocodes
        geocoded = sum(1 for a in articles 
                      if a.get('stage3', {}).get('primary_location', {}).get('lat') is not None)
        geocode_rate = geocoded / len(articles) * 100 if articles else 0
        logger.info(f"  Articles with coordinates: {geocoded}/{len(articles)} ({geocode_rate:.1f}%)")
    
    # Sample outputs
    logger.info(f"\nSample extractions:")
    for i, article in enumerate(articles[:3], 1):
        stage3 = article.get('stage3', {})
        logger.info(f"\n  Example {i}:")
        logger.info(f"    Primary location: {stage3.get('primary_location', {}).get('name', 'N/A')}")
        if stage3.get('primary_location', {}).get('lat'):
            logger.info(f"    Coordinates: ({stage3['primary_location']['lat']:.4f}, "
                       f"{stage3['primary_location']['lon']:.4f})")
        logger.info(f"    Location reasoning: {stage3.get('primary_location', {}).get('reasoning', 'N/A')[:80]}...")
        logger.info(f"    Flood date: {stage3.get('flood_date', {}).get('date', 'N/A')} "
                   f"(precision: {stage3.get('flood_date', {}).get('precision', 'N/A')})")
        logger.info(f"    Date reasoning: {stage3.get('flood_date', {}).get('reasoning', 'N/A')[:80]}...")
    
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3 COMPLETE")
    logger.info("=" * 70)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Stage 3: Extract locations and dates')
    parser.add_argument('--input', type=str, help='Input file (Stage 2 output)')
    parser.add_argument('--output', type=str, help='Output file (Stage 3 output)')
    parser.add_argument('--model', type=str, help='Path to trained model')
    parser.add_argument('--no-geocode', action='store_true', help='Skip geocoding')
    parser.add_argument('--mapbox-token', type=str, help='Mapbox API token')
    parser.add_argument('--batch-interval', type=int, default=1000, help='Save interval')
    
    args = parser.parse_args()
    
    # Set default paths
    input_file = Path(args.input) if args.input else PROJ_ROOT / 'data' / 'processed' / 'stage2_output.json'
    output_file = Path(args.output) if args.output else PROJ_ROOT / 'data' / 'processed' / 'stage3_output.json'
    model_file = Path(args.model) if args.model else PROJ_ROOT / 'stage3' / 'models' / 'optimized_extractor.json'
    
    # Check input exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}")
        print("Please run Stage 2 first or specify --input")
        return
    
    # Create output directory
    output_file.parent.mkdir(exist_ok=True, parents=True)
    
    # Run extraction
    run_extraction(
        input_file=input_file,
        output_file=output_file,
        model_file=model_file,
        geocode=not args.no_geocode,
        mapbox_token=args.mapbox_token,
        batch_save_interval=args.batch_interval
    )


if __name__ == '__main__':
    main()
