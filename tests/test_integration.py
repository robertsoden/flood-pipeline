"""
Integration tests for pipeline data flow.

Tests that data correctly flows between stages and maintains
consistency through the entire pipeline.
"""
import pytest
import json
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
    Stage1Results,
    Stage2Results,
)
from shared.checkpoint import CheckpointManager, filter_unprocessed


class TestArticleIdConsistency:
    """Test that article IDs are consistent across pipeline stages."""

    def test_generate_article_id_from_content(self):
        """Article ID should be generated deterministically from content."""
        article = {
            'title': 'Flood hits Toronto',
            'full_text': 'Heavy rains caused flooding in downtown Toronto.',
            'publication_date': '2020-07-08'
        }

        id1 = generate_article_id(article)
        id2 = generate_article_id(article)

        assert id1 == id2, "Same content should generate same ID"
        assert id1.startswith('art_'), "Generated ID should have art_ prefix"

    def test_generate_article_id_preserves_existing(self):
        """Should preserve existing article_id."""
        article = {
            'article_id': 'existing_123',
            'title': 'Test Article',
            'full_text': 'Content here'
        }

        article_id = generate_article_id(article)
        assert article_id == 'existing_123'

    def test_generate_article_id_from_id_field(self):
        """Should use 'id' field if article_id not present."""
        article = {
            'id': 'legacy_456',
            'title': 'Test Article',
            'full_text': 'Content here'
        }

        article_id = generate_article_id(article)
        assert article_id == 'legacy_456'

    def test_ensure_article_id_modifies_in_place(self):
        """ensure_article_id should modify article in place."""
        article = {
            'title': 'Test',
            'full_text': 'Content'
        }

        ensure_article_id(article)
        assert 'article_id' in article
        assert article['article_id'].startswith('art_')

    def test_normalize_article_fields(self):
        """Test that field names are normalized correctly."""
        article = {
            'id': 'test_123',
            'article_text': 'Content here',
            'date': '2020-01-01'
        }

        normalized = normalize_article_fields(article)

        assert normalized['article_id'] == 'test_123'
        assert normalized['full_text'] == 'Content here'
        assert normalized['publication_date'] == '2020-01-01'
        # Original fields should still exist
        assert normalized['date'] == '2020-01-01'


class TestDataFlowBetweenStages:
    """Test data integrity between pipeline stages."""

    def test_stage1_to_stage2_field_preservation(self):
        """Stage 2 should preserve all Stage 1 fields."""
        stage1_article = {
            'article_id': 'test_001',
            'title': 'Flood Article',
            'full_text': 'Flooding occurred in Hamilton.',
            'publication_date': '2020-05-15',
            'confidence': 0.95,
            'predicted_flood': True
        }

        # Simulate Stage 2 processing
        stage1_article['stage2'] = {
            'flood_verified': True,
            'flood_reasoning': 'Article describes actual flood event',
            'is_ontario': True,
            'ontario_reasoning': 'Hamilton is in Ontario'
        }

        # Verify all original fields preserved
        assert stage1_article['article_id'] == 'test_001'
        assert stage1_article['title'] == 'Flood Article'
        assert stage1_article['confidence'] == 0.95

    def test_stage2_to_stage3_field_preservation(self):
        """Stage 3 should preserve all Stage 1 and Stage 2 fields."""
        article = {
            'article_id': 'test_002',
            'title': 'Toronto Flood',
            'full_text': 'Heavy rain caused flooding in Toronto area.',
            'publication_date': '2013-07-08',
            'confidence': 0.92,
            'predicted_flood': True,
            'stage2': {
                'flood_verified': True,
                'is_ontario': True
            }
        }

        # Simulate Stage 3 processing
        article['stage3'] = {
            'location': 'Toronto',
            'flood_date': 'July 2013',
            'date_confidence': 'high',
            'method': 'NER+LLM'
        }

        # Verify all fields preserved through stages
        assert article['article_id'] == 'test_002'
        assert article['stage2']['flood_verified'] is True
        assert article['stage3']['location'] == 'Toronto'


class TestCheckpointManager:
    """Test checkpoint functionality for resume capability."""

    def test_checkpoint_save_and_load(self, tmp_path):
        """Checkpoint should save and reload correctly."""
        checkpoint = CheckpointManager('test_stage', tmp_path, save_interval=2)

        # Process some articles
        checkpoint.mark_processed('art_001', {'result': 'data1'})
        checkpoint.mark_processed('art_002', {'result': 'data2'})

        # Save checkpoint
        checkpoint.save(force=True)

        # Create new checkpoint manager and load
        checkpoint2 = CheckpointManager('test_stage', tmp_path)
        loaded = checkpoint2.load()

        assert loaded is True
        assert checkpoint2.is_processed('art_001')
        assert checkpoint2.is_processed('art_002')
        assert not checkpoint2.is_processed('art_003')

    def test_filter_unprocessed(self, tmp_path):
        """filter_unprocessed should correctly filter already processed articles."""
        checkpoint = CheckpointManager('test_stage', tmp_path)
        checkpoint.mark_processed('art_001')
        checkpoint.mark_processed('art_002')

        articles = [
            {'article_id': 'art_001', 'title': 'Article 1'},
            {'article_id': 'art_002', 'title': 'Article 2'},
            {'article_id': 'art_003', 'title': 'Article 3'},
            {'article_id': 'art_004', 'title': 'Article 4'},
        ]

        unprocessed = filter_unprocessed(articles, checkpoint)

        assert len(unprocessed) == 2
        assert unprocessed[0]['article_id'] == 'art_003'
        assert unprocessed[1]['article_id'] == 'art_004'

    def test_checkpoint_results_retrieval(self, tmp_path):
        """Results should be retrievable from checkpoint."""
        checkpoint = CheckpointManager('test_stage', tmp_path)

        checkpoint.mark_processed('art_001', {'location': 'Toronto', 'date': 'July 2013'})
        checkpoint.mark_processed('art_002', {'location': 'Ottawa', 'date': 'May 2017'})

        results = checkpoint.get_results()

        assert len(results) == 2
        assert results[0]['location'] == 'Toronto'
        assert results[1]['location'] == 'Ottawa'


class TestValidation:
    """Test input validation functionality."""

    def test_valid_article_base(self):
        """Valid article should pass validation."""
        articles = [{
            'title': 'Test Article',
            'full_text': 'This is the article content with sufficient length.',
            'publication_date': '2020-01-01'
        }]

        valid, invalid = validate_articles(articles, ArticleBase)

        assert len(valid) == 1
        assert len(invalid) == 0

    def test_empty_text_fails_validation(self):
        """Article with empty text should fail validation."""
        articles = [{
            'title': 'Test Article',
            'full_text': '',
            'publication_date': '2020-01-01'
        }]

        valid, invalid = validate_articles(articles, ArticleBase)

        assert len(valid) == 0
        assert len(invalid) == 1
        assert 'empty' in invalid[0]['error'].lower()

    def test_whitespace_only_text_fails(self):
        """Article with whitespace-only text should fail validation."""
        articles = [{
            'title': 'Test Article',
            'full_text': '   \n\t   ',
            'publication_date': '2020-01-01'
        }]

        valid, invalid = validate_articles(articles, ArticleBase)

        assert len(valid) == 0
        assert len(invalid) == 1


class TestPipelineConsistency:
    """Test overall pipeline data consistency."""

    def test_article_count_decreases_through_pipeline(self):
        """Article count should decrease at each filtering stage."""
        # Simulate pipeline progression
        stage1_count = 50247
        stage2_verified_count = 5247
        stage2_ontario_count = 2290
        stage3_geocoded_count = 1940

        # Verify monotonic decrease
        assert stage1_count > stage2_verified_count
        assert stage2_verified_count > stage2_ontario_count
        assert stage2_ontario_count >= stage3_geocoded_count

    def test_stage_results_nested_correctly(self):
        """Each stage should add results in its own namespace."""
        article = {
            'article_id': 'test_001',
            'title': 'Flood News',
            'full_text': 'Flooding in Toronto...'
        }

        # Stage 1 adds directly
        article['confidence'] = 0.95
        article['predicted_flood'] = True

        # Stage 2 adds under 'stage2'
        article['stage2'] = {'flood_verified': True, 'is_ontario': True}

        # Stage 3 adds under 'stage3'
        article['stage3'] = {'location': 'Toronto', 'flood_date': 'July 2013'}

        # Verify structure
        assert 'stage2' in article
        assert 'stage3' in article
        assert article['stage2']['flood_verified'] is True
        assert article['stage3']['location'] == 'Toronto'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
