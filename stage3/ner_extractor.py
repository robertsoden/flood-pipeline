"""
NER-based location and date extraction using spaCy + regex patterns.
Fast, local, no API calls needed.
"""
import re
import spacy
from typing import Tuple, List, Optional
from datetime import datetime

# Load spaCy model (small English model)
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Warning: spaCy model not found. Using regex-only mode.")
    nlp = None

# Common Ontario place names (cities, towns, regions, rivers)
ONTARIO_PLACES = {
    'Toronto', 'Ottawa', 'Mississauga', 'Brampton', 'Hamilton', 'London',
    'Markham', 'Vaughan', 'Kitchener', 'Windsor', 'Richmond Hill', 'Oakville',
    'Burlington', 'Oshawa', 'Barrie', 'St. Catharines', 'Cambridge', 'Kingston',
    'Guelph', 'Whitby', 'Thunder Bay', 'Waterloo', 'Sudbury', 'Brantford',
    'Pickering', 'Niagara Falls', 'Peterborough', 'Sault Ste. Marie', 'Sarnia',
    'North Bay', 'Cornwall', 'Belleville', 'Welland', 'Timmins', 'Chatham',
    'Vaughan', 'Scarborough', 'Etobicoke', 'North York',

    # Regions
    'Muskoka', 'Haliburton', 'Kawartha', 'Niagara', 'Essex', 'Kent', 'Elgin',
    'Oxford', 'Perth', 'Huron', 'Bruce', 'Grey', 'Simcoe', 'York', 'Durham',
    'Peel', 'Halton', 'Waterloo', 'Wellington', 'Dufferin', 'Peterborough',

    # Rivers
    'Grand River', 'Thames River', 'Credit River', 'Humber River', 'Don River',
    'Rouge River', 'Trent River', 'Moira River', 'Ottawa River', 'Rideau River',
    'Speed River', 'Saugeen River', 'Maitland River', 'Ausable River',

    # Lakes
    'Lake Ontario', 'Lake Erie', 'Lake Huron', 'Lake Superior', 'Lake Simcoe',
    'Lake Nipissing', 'Rice Lake', 'Lake of Bays', 'Georgian Bay'
}

# Month names for date extraction
MONTHS = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December',
    'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
    'Jul', 'Aug', 'Sep', 'Sept', 'Oct', 'Nov', 'Dec'
]

MONTH_PATTERN = '|'.join(MONTHS)

def extract_locations(text: str) -> List[str]:
    """
    Extract Ontario place names from text using spaCy NER + pattern matching.

    Returns list of locations found, ordered by first appearance.
    """
    locations = []

    # Method 1: Use spaCy NER if available
    if nlp:
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ('GPE', 'LOC', 'FAC'):  # Geo-political entity, location, facility
                # Check if it's a known Ontario place
                if ent.text in ONTARIO_PLACES:
                    locations.append((ent.start_char, ent.text))
                # Or check case-insensitive match
                elif any(place.lower() == ent.text.lower() for place in ONTARIO_PLACES):
                    matching_place = next(p for p in ONTARIO_PLACES if p.lower() == ent.text.lower())
                    locations.append((ent.start_char, matching_place))

    # Method 2: Pattern matching for known Ontario places (fallback or supplement)
    text_lower = text.lower()
    for place in ONTARIO_PLACES:
        pattern = r'\b' + re.escape(place.lower()) + r'\b'
        if re.search(pattern, text_lower):
            match = re.search(pattern, text_lower)
            # Avoid duplicates
            if not any(loc[1].lower() == place.lower() for loc in locations):
                locations.append((match.start(), place))

    # Sort by position (earlier mentions more likely to be primary location)
    locations.sort(key=lambda x: x[0])

    return [loc for _, loc in locations]

def extract_dates(text: str) -> List[str]:
    """
    Extract dates in 'Month Year' format from text.

    Returns list of dates found, formatted as 'Month YYYY'.
    """
    dates = []

    # Pattern: Month YYYY (e.g., "May 1997", "April 2013")
    pattern = rf'\b({MONTH_PATTERN})\s+(\d{{4}})\b'
    matches = re.finditer(pattern, text, re.IGNORECASE)

    for match in matches:
        month = match.group(1).capitalize()
        year = match.group(2)

        # Normalize month name (Jan -> January, etc)
        if len(month) == 3:
            month_map = {
                'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
                'Apr': 'April', 'May': 'May', 'Jun': 'June',
                'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
                'Sept': 'September', 'Oct': 'October', 'Nov': 'November',
                'Dec': 'December'
            }
            month = month_map.get(month, month)

        dates.append(f"{month} {year}")

    # Also try: early/late/mid Month YYYY
    pattern2 = rf'\b(early|late|mid)\s+({MONTH_PATTERN})\s+(\d{{4}})\b'
    matches = re.finditer(pattern2, text, re.IGNORECASE)

    for match in matches:
        qualifier = match.group(1).lower()
        month = match.group(2).capitalize()
        year = match.group(3)

        # Normalize month
        if len(month) == 3:
            month_map = {
                'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
                'Apr': 'April', 'May': 'May', 'Jun': 'June',
                'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
                'Sept': 'September', 'Oct': 'October', 'Nov': 'November',
                'Dec': 'December'
            }
            month = month_map.get(month, month)

        dates.append(f"{qualifier} {month} {year}")

    # Deduplicate while preserving order
    seen = set()
    unique_dates = []
    for date in dates:
        if date not in seen:
            seen.add(date)
            unique_dates.append(date)

    return unique_dates

def extract_location_date(title: str, article_text: str) -> Tuple[str, str]:
    """
    Extract primary location and flood date from article.

    Uses simple heuristics:
    - First mentioned location is usually the primary one
    - First mentioned date is usually the flood date

    Returns:
        (location, flood_date) - strings, or empty if not found
    """
    # Combine title and text, with title getting priority
    combined = title + " " + article_text

    # Extract candidates
    locations = extract_locations(combined)
    dates = extract_dates(combined)

    # Pick first location (most likely primary)
    location = locations[0] if locations else ""

    # Pick first date (most likely flood date, not publication date)
    flood_date = dates[0] if dates else ""

    return location, flood_date

# Test function
if __name__ == "__main__":
    # Test with sample text
    test_text = """
    Flooding in the Grand River region near Kitchener-Waterloo in May 2000
    caused extensive damage. The flood occurred in early May 2000 after
    heavy rainfall. Residents of Cambridge were also affected.
    """

    title = "Grand River floods Kitchener region"
    location, date = extract_location_date(title, test_text)

    print(f"Location: {location}")
    print(f"Date: {date}")

    # Show all candidates
    print(f"\nAll locations found: {extract_locations(test_text)}")
    print(f"All dates found: {extract_dates(test_text)}")
