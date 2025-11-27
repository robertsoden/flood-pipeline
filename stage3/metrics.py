"""
Stage 3 Metrics: Evaluate location and date extraction with fuzzy matching
"""
import dspy
import re
from difflib import SequenceMatcher

def normalize_location(loc):
    """Normalize location string for comparison"""
    if not loc:
        return ""
    # Convert to lowercase
    loc = loc.lower().strip()
    # Remove common suffixes and punctuation
    loc = re.sub(r',?\s*(ont\.|ontario|canada)\s*', ' ', loc)
    loc = re.sub(r'[,;.]', ' ', loc)
    # Normalize whitespace
    loc = ' '.join(loc.split())
    return loc

def extract_date_components(date_str):
    """Extract year, month from various date formats"""
    if not date_str:
        return None, None

    date_str = date_str.lower().strip()

    # Extract 4-digit year
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    year = year_match.group(1) if year_match else None

    # Extract month (name or number)
    month = None
    month_names = ['january', 'february', 'march', 'april', 'may', 'june',
                   'july', 'august', 'september', 'october', 'november', 'december']
    month_abbr = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    for i, name in enumerate(month_names):
        if name in date_str:
            month = str(i + 1).zfill(2)
            break

    if not month:
        for i, abbr in enumerate(month_abbr):
            if abbr in date_str:
                month = str(i + 1).zfill(2)
                break

    # Try to find MM format or M format
    if not month:
        month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', date_str)
        if month_match:
            month = month_match.group(1).zfill(2)

    return year, month

def fuzzy_location_match(pred, truth):
    """
    Fuzzy match locations with multiple strategies.
    Returns score between 0 and 1.
    """
    # Handle multi-location ground truth (semicolon-separated)
    if ';' in truth:
        # Try matching against any of the locations
        truth_locs = [loc.strip() for loc in truth.split(';')]
        scores = [fuzzy_location_match(pred, loc) for loc in truth_locs]
        return max(scores) if scores else 0.0

    pred_norm = normalize_location(pred)
    truth_norm = normalize_location(truth)

    # Handle empty normalized values (e.g., "Canada" becomes empty)
    # Give partial credit if prediction has content
    if not truth_norm:
        if pred_norm:
            return 0.5  # Some location vs vague ground truth
        return 0.0

    if not pred_norm:
        return 0.0

    # Exact match after normalization
    if pred_norm == truth_norm:
        return 1.0

    # Substring match (either direction)
    if pred_norm in truth_norm or truth_norm in pred_norm:
        return 0.9

    # Token overlap (for compound locations like "White River" vs "White River, Ontario")
    pred_tokens = set(pred_norm.split())
    truth_tokens = set(truth_norm.split())

    # Remove common stop words
    stop_words = {'the', 'of', 'and', 'in', 'at', 'on', 'near'}
    pred_tokens -= stop_words
    truth_tokens -= stop_words

    if pred_tokens and truth_tokens:
        overlap = len(pred_tokens & truth_tokens)
        union = len(pred_tokens | truth_tokens)
        if overlap > 0:
            jaccard = overlap / union
            if jaccard >= 0.5:  # At least 50% token overlap
                return 0.7 + (jaccard - 0.5) * 0.6  # Scale 0.7-1.0

    # Fuzzy string matching (for typos, slight variations)
    similarity = SequenceMatcher(None, pred_norm, truth_norm).ratio()
    if similarity >= 0.8:
        return 0.6 + (similarity - 0.8) * 2  # Scale 0.6-1.0
    elif similarity >= 0.6:
        return 0.3 + (similarity - 0.6) * 1.5  # Scale 0.3-0.6

    return 0.0

def fuzzy_date_match(pred, truth):
    """
    Fuzzy match dates - accept if year+month match or just year matches.
    Returns score between 0 and 1.
    """
    pred_year, pred_month = extract_date_components(pred)
    truth_year, truth_month = extract_date_components(truth)

    if not pred_year or not truth_year:
        # One or both missing - can't match
        return 0.0

    # Year must match
    if pred_year != truth_year:
        return 0.0

    # Year matches - give base score
    score = 0.6

    # Bonus if month also matches
    if pred_month and truth_month and pred_month == truth_month:
        score = 1.0
    elif pred_month and truth_month:
        # Month extracted but doesn't match
        score = 0.7

    return score

def location_extraction_metric(example, prediction, trace=None):
    """
    Metric for evaluating location and date extraction with fuzzy matching.

    For labeled data with known locations/dates:
    - Fuzzy matches location (handles variations, multi-location ground truth)
    - Fuzzy matches dates (accepts year+month, or just year)

    Returns score between 0 and 1.
    """
    score = 0.0

    # Check if we have ground truth
    if not hasattr(example, 'location') or not example.location:
        # No ground truth - can't evaluate, skip this example
        return None

    # Location matching with fuzzy matching (70% of score)
    if hasattr(prediction, 'location') and prediction.location:
        pred_loc = prediction.location
        true_loc = example.location

        # Skip if prediction is explicitly "unknown"
        if pred_loc.lower() in ['unknown', 'not specified', 'none', 'n/a']:
            score += 0.0
        else:
            loc_score = fuzzy_location_match(pred_loc, true_loc)
            score += loc_score * 0.7

    # Date matching with fuzzy matching (30% of score)
    if hasattr(example, 'flood_date') and example.flood_date:
        if hasattr(prediction, 'flood_date') and prediction.flood_date:
            pred_date = prediction.flood_date
            true_date = example.flood_date

            # Skip if prediction is explicitly "unknown"
            if pred_date.lower() in ['unknown', 'not specified', 'date not specified', 'none', 'n/a']:
                score += 0.0
            else:
                date_score = fuzzy_date_match(pred_date, true_date)
                score += date_score * 0.3
    else:
        # No date ground truth, give partial credit
        score += 0.15

    return score
