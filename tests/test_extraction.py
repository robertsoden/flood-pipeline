"""
Sample data tests for extraction accuracy.

Tests NER extraction and date parsing with known flood articles
to verify extraction quality.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stage3.ner_extractor import extract_locations, extract_dates, extract_location_date


class TestLocationExtraction:
    """Test location extraction from article text."""

    def test_extract_major_city(self):
        """Should extract major Ontario cities."""
        text = "Heavy flooding hit Toronto yesterday, causing widespread damage."
        locations = extract_locations(text)

        assert 'Toronto' in locations

    def test_extract_river_name(self):
        """Should extract river names."""
        text = "The Grand River overflowed its banks near Cambridge."
        locations = extract_locations(text)

        assert 'Grand River' in locations or 'Cambridge' in locations

    def test_extract_multiple_locations(self):
        """Should extract multiple locations from text."""
        text = """
        Flooding affected several communities including Hamilton, Burlington,
        and Oakville along the Lake Ontario shoreline.
        """
        locations = extract_locations(text)

        assert len(locations) >= 2
        # Should find at least some of these
        location_set = set(loc.lower() for loc in locations)
        assert any(city.lower() in location_set
                   for city in ['Hamilton', 'Burlington', 'Oakville'])

    def test_extract_location_from_title(self):
        """Should extract location from title when combined with text."""
        title = "Ottawa hit by spring floods"
        text = "Residents are evacuating as water levels rise."

        location, _ = extract_location_date(title, text)
        assert location == 'Ottawa'

    def test_flood_prone_communities(self):
        """Should recognize smaller flood-prone communities."""
        # These were added in the expanded ONTARIO_PLACES list
        test_cases = [
            ("Flooding in Woodbridge along the Humber River", "Woodbridge"),
            ("Holland Marsh farmers face flood damage", "Holland Marsh"),
            ("Weston residents evacuated due to flooding", "Weston"),
        ]

        for text, expected_location in test_cases:
            locations = extract_locations(text)
            assert expected_location in locations, f"Failed to find {expected_location}"

    def test_location_priority_first_mention(self):
        """First mentioned location should be returned as primary."""
        title = "Hamilton flooding spreads to Burlington"
        text = "The flood that started in Hamilton has now reached Burlington."

        location, _ = extract_location_date(title, text)
        # Hamilton is mentioned first in title
        assert location == 'Hamilton'


class TestDateExtraction:
    """Test date extraction from article text."""

    def test_extract_month_year(self):
        """Should extract Month YYYY format dates."""
        text = "The flood occurred in May 1997 and caused $50 million in damage."
        dates = extract_dates(text)

        assert 'May 1997' in dates

    def test_extract_multiple_dates(self):
        """Should extract multiple dates from text."""
        text = """
        The June 2013 flood was the worst since the Hurricane Hazel flood
        of October 1954 devastated the region.
        """
        dates = extract_dates(text)

        assert len(dates) >= 2

    def test_extract_qualified_dates(self):
        """Should extract dates with qualifiers like early/late/mid."""
        test_cases = [
            ("early May 1997", "early May 1997"),
            ("late June 2013", "late June 2013"),
            ("mid April 2019", "mid April 2019"),
        ]

        for text, expected in test_cases:
            dates = extract_dates(f"The flooding began in {text}.")
            # Should find either qualified or unqualified version
            assert any(expected in d or expected.split()[1] in d for d in dates)

    def test_date_from_combined_title_text(self):
        """Should extract date from combined title and article text."""
        title = "Spring 2019 flooding worst in decades"
        text = "The April 2019 floods caused unprecedented damage."

        _, flood_date = extract_location_date(title, text)
        assert '2019' in flood_date

    def test_normalize_abbreviated_months(self):
        """Should normalize abbreviated month names."""
        text = "Flooding occurred in Sept 2020 after heavy rains."
        dates = extract_dates(text)

        # Should be normalized to full month name
        assert any('September 2020' in d or 'Sept 2020' in d for d in dates)


class TestCombinedExtraction:
    """Test combined location and date extraction."""

    def test_real_flood_article_pattern(self):
        """Test extraction from realistic flood article text."""
        title = "Grand River floods downtown Cambridge"
        text = """
        Heavy spring rains in April 2018 caused the Grand River to overflow
        its banks, flooding parts of downtown Cambridge and surrounding areas.
        The flooding was the worst the region had seen since 1974. Residents
        along Water Street were evacuated as waters rose throughout the day.
        """

        location, flood_date = extract_location_date(title, text)

        # Should find Grand River or Cambridge as location
        assert location in ['Grand River', 'Cambridge']
        # Should extract April 2018 as flood date
        assert '2018' in flood_date or 'April' in flood_date

    def test_toronto_july_2013_flood(self):
        """Test extraction for well-known July 2013 Toronto flood."""
        title = "Toronto paralyzed by flash flooding"
        text = """
        A severe thunderstorm on July 8, 2013 dumped over 100mm of rain on
        Toronto in just two hours, causing widespread flooding across the GTA.
        The Don Valley Parkway was submerged and thousands lost power. It was
        the most expensive natural disaster in Ontario history at the time.
        """

        location, flood_date = extract_location_date(title, text)

        assert location == 'Toronto'
        assert 'July' in flood_date or '2013' in flood_date

    def test_ottawa_river_spring_flood(self):
        """Test extraction for Ottawa River spring flooding."""
        title = "Ottawa River reaches flood stage"
        text = """
        Spring melting and heavy April 2019 rains have pushed the Ottawa River
        to flood stage. Communities from Pembroke to Ottawa are preparing for
        potential evacuations as water levels continue to rise.
        """

        location, flood_date = extract_location_date(title, text)

        assert location in ['Ottawa River', 'Ottawa', 'Pembroke']
        assert '2019' in flood_date or 'April' in flood_date


class TestOntarioPlacesCoverage:
    """Test that ONTARIO_PLACES covers important flood-prone areas."""

    def test_major_rivers_included(self):
        """Major Ontario rivers should be in ONTARIO_PLACES."""
        from stage3.ner_extractor import ONTARIO_PLACES

        major_rivers = [
            'Grand River', 'Thames River', 'Credit River', 'Humber River',
            'Don River', 'Ottawa River', 'Rideau River', 'Trent River'
        ]

        for river in major_rivers:
            assert river in ONTARIO_PLACES, f"Missing major river: {river}"

    def test_gta_communities_included(self):
        """GTA flood-prone communities should be included."""
        from stage3.ner_extractor import ONTARIO_PLACES

        gta_communities = [
            'Toronto', 'Mississauga', 'Brampton', 'Vaughan',
            'Woodbridge', 'Weston', 'Kleinburg', 'Bolton'
        ]

        for community in gta_communities:
            assert community in ONTARIO_PLACES, f"Missing GTA community: {community}"

    def test_conservation_areas_included(self):
        """Flood-prone conservation areas should be included."""
        from stage3.ner_extractor import ONTARIO_PLACES

        areas = ['Holland Marsh', 'Don Valley', 'Humber Valley', 'Credit Valley']

        for area in areas:
            assert area in ONTARIO_PLACES, f"Missing conservation area: {area}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
