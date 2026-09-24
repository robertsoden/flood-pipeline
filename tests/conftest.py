"""
Pytest configuration and fixtures for flood_news tests.
"""
import pytest
import sys
from pathlib import Path
import json
import tempfile
import shutil

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_article():
    """Return a sample flood article for testing."""
    return {
        'article_id': 'test_001',
        'title': 'Flooding hits Greater Toronto Area',
        'full_text': '''
        Heavy rainfall on July 8, 2013 caused severe flooding across the Greater
        Toronto Area. The Don Valley Parkway was submerged and GO Transit service
        was suspended. Thousands of commuters were stranded. The storm dumped over
        100mm of rain in just two hours, overwhelming the city's drainage system.
        ''',
        'publication_date': 'Jul 9, 2013'
    }


@pytest.fixture
def sample_articles_batch():
    """Return a batch of sample articles for testing."""
    return [
        {
            'article_id': 'flood_001',
            'title': 'Toronto flooding paralyzes transit',
            'full_text': 'Heavy July 2013 rains flooded the Don Valley Parkway.',
            'publication_date': '2013-07-09'
        },
        {
            'article_id': 'flood_002',
            'title': 'Grand River reaches flood stage',
            'full_text': 'Spring flooding in April 2019 affected Cambridge.',
            'publication_date': '2019-04-20'
        },
        {
            'article_id': 'flood_003',
            'title': 'Ottawa residents evacuate',
            'full_text': 'Ottawa River flooding forces evacuations in May 2017.',
            'publication_date': '2017-05-08'
        },
        {
            'article_id': 'non_flood_001',
            'title': 'Flood insurance rates rise',
            'full_text': 'Insurance companies are raising flood coverage rates.',
            'publication_date': '2020-01-15'
        },
        {
            'article_id': 'non_flood_002',
            'title': 'City flooded with complaints',
            'full_text': 'Residents flooded city hall with complaints about taxes.',
            'publication_date': '2020-03-01'
        }
    ]


@pytest.fixture
def stage1_output():
    """Return sample Stage 1 BERT output."""
    return [
        {
            'article_id': 'bert_001',
            'title': 'Flooding in Hamilton',
            'full_text': 'Heavy rains caused flooding in downtown Hamilton.',
            'publication_date': '2020-05-15',
            'confidence': 0.95,
            'predicted_flood': True
        },
        {
            'article_id': 'bert_002',
            'title': 'Weather report',
            'full_text': 'Rain expected this week across Ontario.',
            'publication_date': '2020-05-14',
            'confidence': 0.15,
            'predicted_flood': False
        }
    ]


@pytest.fixture
def stage2_output():
    """Return sample Stage 2 output with flood verification."""
    return [
        {
            'article_id': 'stage2_001',
            'title': 'Toronto flood causes chaos',
            'full_text': 'Flash flooding in Toronto submerged vehicles.',
            'publication_date': '2013-07-08',
            'confidence': 0.92,
            'predicted_flood': True,
            'stage2': {
                'flood_verified': True,
                'flood_reasoning': 'Article describes actual flood event with impacts.',
                'is_ontario': True,
                'ontario_reasoning': 'Toronto is in Ontario, Canada.'
            }
        }
    ]


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Create a temporary directory for checkpoint testing."""
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    yield checkpoint_dir
    # Cleanup handled by pytest tmp_path fixture


@pytest.fixture
def temp_results_dir(tmp_path):
    """Create a temporary directory for results testing."""
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    yield results_dir


@pytest.fixture
def mock_stage1_results(temp_results_dir, stage1_output):
    """Create mock Stage 1 results file."""
    output_file = temp_results_dir / 'predicted_floods.json'
    with open(output_file, 'w') as f:
        json.dump(stage1_output, f)
    return output_file


@pytest.fixture
def mock_ner_results(temp_results_dir):
    """Create mock NER extraction results."""
    ner_results = [
        {
            'article_id': 'ner_001',
            'title': 'Hamilton flood damages homes',
            'full_text': 'The May 2020 flooding in Hamilton caused widespread damage.',
            'publication_date': '2020-05-18',
            'stage2': {'flood_verified': True, 'is_ontario': True},
            'stage3': {
                'location': 'Hamilton',
                'flood_date': 'May 2020',
                'method': 'NER'
            }
        },
        {
            'article_id': 'ner_002',
            'title': 'Credit River overflows',
            'full_text': 'Spring rains caused the Credit River to overflow in April.',
            'publication_date': '2019-04-25',
            'stage2': {'flood_verified': True, 'is_ontario': True},
            'stage3': {
                'location': 'Credit River',
                'flood_date': '',
                'method': 'NER'
            }
        }
    ]

    output_file = temp_results_dir / 'stage3_extracted_ner.json'
    with open(output_file, 'w') as f:
        json.dump(ner_results, f)
    return output_file


# Skip markers for slow tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_api: marks tests that require external API access"
    )
