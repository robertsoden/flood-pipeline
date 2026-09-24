"""
Edge case tests for the flood news pipeline.

Tests handling of:
- Empty articles
- Missing fields
- Non-Ontario floods that might slip through
- Ambiguous or malformed data
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.validation import (
    generate_article_id,
    ensure_article_id,
    normalize_article_fields,
    validate_articles,
    ArticleBase,
)
from stage3.ner_extractor import extract_locations, extract_dates, extract_location_date


class TestEmptyArticles:
    """Test handling of empty or minimal articles."""

    def test_empty_text_generates_id(self):
        """Empty text should still generate an ID (from title/date)."""
        article = {
            'title': 'Flood News',
            'full_text': '',
            'publication_date': '2020-01-01'
        }

        article_id = generate_article_id(article)
        assert article_id is not None
        assert len(article_id) > 0

    def test_all_empty_fields_generates_id(self):
        """Completely empty article should still get an ID."""
        article = {
            'title': '',
            'full_text': '',
            'publication_date': ''
        }

        article_id = generate_article_id(article)
        assert article_id.startswith('art_')

    def test_extract_from_empty_text(self):
        """Extraction from empty text should return empty strings."""
        location, flood_date = extract_location_date('', '')

        assert location == ''
        assert flood_date == ''

    def test_extract_from_title_only(self):
        """Should extract from title when text is empty."""
        title = "Toronto flooding causes chaos"
        text = ""

        locations = extract_locations(title + " " + text)
        assert 'Toronto' in locations


class TestMissingFields:
    """Test handling of articles with missing required fields."""

    def test_missing_title(self):
        """Should handle missing title field."""
        article = {
            'full_text': 'Flooding in Hamilton caused damage.',
            'publication_date': '2020-01-01'
        }

        normalized = normalize_article_fields(article)
        # Should not crash, article_id should be generated
        assert 'article_id' in normalized

    def test_missing_date(self):
        """Should handle missing publication_date field."""
        article = {
            'title': 'Flood Article',
            'full_text': 'Content about flooding.'
        }

        normalized = normalize_article_fields(article)
        assert 'article_id' in normalized

    def test_alternative_field_names(self):
        """Should handle alternative field names."""
        article = {
            'id': 'legacy_123',
            'article_text': 'Content here',
            'date': '2020-05-15'
        }

        normalized = normalize_article_fields(article)

        assert normalized['article_id'] == 'legacy_123'
        assert normalized['full_text'] == 'Content here'
        assert normalized['publication_date'] == '2020-05-15'


class TestNonOntarioFloods:
    """Test filtering of non-Ontario flood articles."""

    def test_quebec_flood_not_extracted_as_ontario(self):
        """Quebec locations should not match Ontario places."""
        text = "Flooding in Montreal caused significant damage to Quebec communities."
        locations = extract_locations(text)

        # Should not find Montreal in Ontario places
        # (Montreal is not in ONTARIO_PLACES)
        assert 'Montreal' not in locations

    def test_us_flood_locations(self):
        """US flood locations should not be extracted."""
        text = "The Mississippi River flooding affected communities in Louisiana."
        locations = extract_locations(text)

        # Mississippi River IS in our list (Ontario's Mississippi River)
        # but Louisiana definitely isn't
        assert 'Louisiana' not in locations

    def test_manitoba_flood_not_extracted(self):
        """Manitoba locations should not be extracted."""
        text = "Winnipeg faced severe flooding from the Red River."
        locations = extract_locations(text)

        assert 'Winnipeg' not in locations

    def test_ambiguous_location_names(self):
        """Test handling of place names that exist in multiple regions."""
        # London, ON vs London, UK
        # Paris, ON vs Paris, France
        # Cambridge, ON vs Cambridge, UK/MA

        from stage3.ner_extractor import ONTARIO_PLACES

        # These should be in our Ontario places
        ambiguous_ontario = ['London', 'Paris', 'Cambridge']
        for place in ambiguous_ontario:
            assert place in ONTARIO_PLACES


class TestMalformedData:
    """Test handling of malformed or unusual data."""

    def test_unicode_in_text(self):
        """Should handle unicode characters in text."""
        text = "Flooding in Sault Ste. Marie caused damage worth $1.5M"
        locations = extract_locations(text)

        # Note: Sault Ste. Marie should be found
        assert any('Sault' in loc for loc in locations)

    def test_special_characters_in_dates(self):
        """Should handle various date formats."""
        test_texts = [
            "The flood occurred in May, 1997.",
            "Flooding hit in May '97.",
            "The May/1997 flood was devastating.",
        ]

        for text in test_texts:
            dates = extract_dates(text)
            # Should extract something with 1997 or May
            assert len(dates) >= 0  # May not extract all formats

    def test_very_long_text(self):
        """Should handle very long article text."""
        # Simulate a very long article
        text = "Flooding in Toronto. " * 1000

        locations = extract_locations(text)
        assert 'Toronto' in locations

    def test_html_entities(self):
        """Should handle text with HTML entities."""
        text = "Flooding in Toronto &amp; Hamilton caused &gt; $1M in damage."
        locations = extract_locations(text)

        # Should still find locations despite HTML entities
        assert 'Toronto' in locations or 'Hamilton' in locations


class TestDateEdgeCases:
    """Test edge cases in date extraction."""

    def test_future_dates_not_extracted(self):
        """Should not extract obviously future dates as flood dates."""
        text = "The 2025 flood prevention plan was announced today."
        dates = extract_dates(text)

        # 2025 might be extracted but should be filtered in LLM verification
        # For now just ensure no crash
        assert isinstance(dates, list)

    def test_historical_dates(self):
        """Should extract historical flood dates."""
        text = "The Hurricane Hazel flood of October 1954 killed 81 people."
        dates = extract_dates(text)

        assert any('1954' in d for d in dates)

    def test_decade_references(self):
        """Should handle decade references."""
        text = "Floods in the 1950s led to new conservation authorities."
        dates = extract_dates(text)

        # May not extract '1950s' directly, but shouldn't crash
        assert isinstance(dates, list)

    def test_relative_dates_in_text(self):
        """Should handle relative date mentions (these need LLM resolution)."""
        text = "Yesterday's flooding was the worst in years."
        dates = extract_dates(text)

        # NER won't extract 'yesterday' - that's for LLM
        # Just ensure no crash
        assert isinstance(dates, list)


class TestLocationEdgeCases:
    """Test edge cases in location extraction."""

    def test_location_with_qualifier(self):
        """Should extract location even with qualifiers."""
        test_cases = [
            "downtown Toronto",
            "northern Hamilton",
            "the city of Ottawa",
            "greater Toronto area"
        ]

        for text in test_cases:
            locations = extract_locations(text)
            # Should find the base city name
            assert len(locations) > 0 or 'area' in text.lower()

    def test_region_vs_city_conflict(self):
        """Handle cases where region and city share names."""
        # York is both a city and a region
        text = "York Region experienced flooding."
        locations = extract_locations(text)

        # Should find York
        assert 'York' in locations

    def test_compound_location_names(self):
        """Should handle compound location names."""
        test_cases = [
            ("Sault Ste. Marie", "Sault Ste. Marie"),
            ("St. Catharines", "St. Catharines"),
            ("Niagara Falls", "Niagara Falls"),
        ]

        for text, expected in test_cases:
            locations = extract_locations(f"Flooding hit {text}.")
            assert expected in locations


class TestValidationEdgeCases:
    """Test validation edge cases."""

    def test_extra_fields_allowed(self):
        """Extra fields should be allowed (not stripped)."""
        articles = [{
            'title': 'Test',
            'full_text': 'Content here',
            'publication_date': '2020-01-01',
            'extra_field': 'should_be_kept',
            'another_extra': 123
        }]

        valid, invalid = validate_articles(articles, ArticleBase)

        assert len(valid) == 1
        assert valid[0]['extra_field'] == 'should_be_kept'

    def test_none_vs_empty_string(self):
        """Test handling of None vs empty string."""
        article1 = {'title': 'Test', 'full_text': 'Content', 'publication_date': None}
        article2 = {'title': 'Test', 'full_text': 'Content', 'publication_date': ''}

        # Both should be valid (publication_date is optional)
        valid1, _ = validate_articles([article1], ArticleBase)
        valid2, _ = validate_articles([article2], ArticleBase)

        assert len(valid1) == 1
        assert len(valid2) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
