"""
Evaluation metrics for Stage 3: Location and Date Extraction

Metrics to evaluate the quality of extracted location and date information.
"""
import dspy
from typing import List


def location_extraction_metric(example: dspy.Example, prediction: dspy.Example, trace=None) -> float:
    """
    Evaluate location extraction quality.

    Uses fuzzy matching since exact string match is too strict for location names
    (e.g., "Thames River" vs "Thames river" vs "the Thames River").

    Returns:
        1.0 if locations match (case-insensitive, normalized)
        0.0 otherwise
    """
    # Normalize locations for comparison
    def normalize_location(loc: str) -> str:
        """Normalize location string for comparison"""
        if not loc:
            return ""
        # Convert to lowercase, remove extra whitespace and common articles
        loc = loc.lower().strip()
        for article in [' the ', ' a ', ' an ']:
            loc = loc.replace(article, ' ')
        # Remove leading/trailing articles
        for article in ['the ', 'a ', 'an ']:
            if loc.startswith(article):
                loc = loc[len(article):]
        # Normalize whitespace
        return ' '.join(loc.split())

    true_location = normalize_location(example.location)
    pred_location = normalize_location(prediction.location)

    # Empty locations
    if not true_location and not pred_location:
        return 1.0
    if not true_location or not pred_location:
        return 0.0

    # Exact match after normalization
    if true_location == pred_location:
        return 1.0

    # Partial match: check if one contains the other (handles "Toronto" vs "Toronto area")
    if true_location in pred_location or pred_location in true_location:
        return 0.8  # Partial credit

    return 0.0


def date_extraction_metric(example: dspy.Example, prediction: dspy.Example, trace=None) -> float:
    """
    Evaluate date extraction quality.

    Handles various date formats (year only, month+year, season+year, etc.)

    Returns:
        1.0 if dates match
        0.5 if years match but months/seasons differ
        0.0 otherwise
    """
    # Normalize dates for comparison
    def normalize_date(date: str) -> str:
        """Normalize date string for comparison"""
        if not date:
            return ""
        # Convert to lowercase, remove extra whitespace
        date = date.lower().strip()
        # Normalize whitespace
        return ' '.join(date.split())

    def extract_year(date: str) -> str:
        """Extract 4-digit year from date string"""
        import re
        match = re.search(r'\b(19\d{2}|20\d{2})\b', date)
        return match.group(1) if match else ""

    true_date = normalize_date(example.flood_date)
    pred_date = normalize_date(prediction.flood_date)

    # Empty dates
    if not true_date and not pred_date:
        return 1.0
    if not true_date or not pred_date:
        return 0.0

    # Exact match
    if true_date == pred_date:
        return 1.0

    # Year-only match (partial credit)
    true_year = extract_year(true_date)
    pred_year = extract_year(pred_date)

    if true_year and pred_year and true_year == pred_year:
        return 0.5  # Partial credit for correct year

    return 0.0


def combined_extraction_metric(example: dspy.Example, prediction: dspy.Example, trace=None) -> float:
    """
    Combined metric for both location and date extraction.

    Returns average score (0-100%) across both extractions.
    """
    location_score = location_extraction_metric(example, prediction, trace)
    date_score = date_extraction_metric(example, prediction, trace)

    # Average the two scores
    combined_score = (location_score + date_score) / 2.0

    # Convert to percentage for display
    return combined_score * 100
