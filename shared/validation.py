"""
Data validation schemas using Pydantic.

Provides validation for article data at pipeline entry points to ensure
data integrity and catch issues early.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from datetime import datetime


class ArticleAnnotations(BaseModel):
    """Annotations for labeled article data (nested format)"""
    flood_mentioned: bool = Field(description="Whether article mentions a flood")
    location: Optional[str] = Field(default="", description="Location of flood if mentioned")
    flood_date: Optional[str] = Field(default="", description="Date of flood if mentioned")
    impacts: Optional[str] = Field(default="", description="Impacts description")
    is_ontario: bool = Field(default=False, description="Whether flood occurred in Ontario")

    model_config = ConfigDict(extra='allow')


class ArticleBase(BaseModel):
    """Base article data structure"""
    title: str = Field(description="Article title")
    full_text: str = Field(min_length=1, description="Full article text content")
    publication_date: Optional[str] = Field(default="", description="Publication date")

    @field_validator('full_text')
    @classmethod
    def validate_text_not_empty(cls, v: str) -> str:
        """Ensure article text is not empty or whitespace only"""
        if not v or not v.strip():
            raise ValueError("Article text cannot be empty")
        return v

    model_config = ConfigDict(extra='allow')


class LabeledArticleNested(ArticleBase):
    """Labeled article with nested annotations (original format)"""
    annotations: ArticleAnnotations = Field(description="Annotation data")
    date: Optional[str] = Field(default="", description="Alternative date field")

    model_config = ConfigDict(extra='allow')


class LabeledArticleFlat(ArticleBase):
    """Labeled article with flat structure (combined format)"""
    article_text: Optional[str] = Field(default=None, description="Alternative text field")
    flood_mentioned: bool = Field(description="Whether article mentions a flood")
    location: Optional[str] = Field(default="", description="Location of flood")
    flood_date: Optional[str] = Field(default="", description="Date of flood")
    impacts: Optional[str] = Field(default="", description="Impacts description")
    is_ontario: bool = Field(default=False, description="Whether flood in Ontario")

    model_config = ConfigDict(extra='allow')


class UnlabeledArticle(ArticleBase):
    """Unlabeled article for inference"""
    model_config = ConfigDict(extra='allow')


class Stage1Results(ArticleBase):
    """Article with Stage 1 BERT results"""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="BERT confidence score")
    predicted_flood: Optional[bool] = Field(default=None, description="BERT prediction")

    model_config = ConfigDict(extra='allow')


class Stage2Results(Stage1Results):
    """Article with Stage 2 results"""
    stage2: Optional[Dict[str, Any]] = Field(default=None, description="Stage 2 verification results")

    @field_validator('stage2')
    @classmethod
    def validate_stage2_structure(cls, v: Optional[Dict]) -> Optional[Dict]:
        """Validate stage2 results structure if present"""
        if v is not None:
            required_fields = ['flood_verified', 'flood_reasoning']
            for field in required_fields:
                if field not in v:
                    raise ValueError(f"Stage 2 results missing required field: {field}")
        return v

    model_config = ConfigDict(extra='allow')


def validate_articles(articles: List[Dict[str, Any]], schema: type[BaseModel]) -> tuple[List[Dict], List[Dict]]:
    """
    Validate a list of articles against a pydantic schema.

    Args:
        articles: List of article dictionaries
        schema: Pydantic model class to validate against

    Returns:
        Tuple of (valid_articles, invalid_articles_with_errors)

    Example:
        >>> valid, invalid = validate_articles(data, LabeledArticleFlat)
        >>> print(f"Valid: {len(valid)}, Invalid: {len(invalid)}")
    """
    valid = []
    invalid = []

    for i, article in enumerate(articles):
        try:
            # Validate using pydantic
            validated = schema.model_validate(article)
            valid.append(article)
        except Exception as e:
            invalid.append({
                'index': i,
                'article': article,
                'error': str(e)
            })

    return valid, invalid


def detect_article_format(article: Dict[str, Any]) -> str:
    """
    Detect the format of an article dictionary.

    Returns:
        'nested' if has 'annotations' field
        'flat' if has 'flood_mentioned' at top level
        'unlabeled' otherwise
    """
    if 'annotations' in article:
        return 'nested'
    elif 'flood_mentioned' in article:
        return 'flat'
    else:
        return 'unlabeled'


def validate_with_auto_detect(articles: List[Dict[str, Any]]) -> tuple[List[Dict], List[Dict], str]:
    """
    Automatically detect format and validate articles.

    Args:
        articles: List of article dictionaries

    Returns:
        Tuple of (valid_articles, invalid_articles, detected_format)

    Example:
        >>> valid, invalid, format_type = validate_with_auto_detect(data)
        >>> print(f"Detected format: {format_type}")
    """
    if not articles:
        return [], [], 'empty'

    # Detect format from first article
    format_type = detect_article_format(articles[0])

    # Select appropriate schema
    schema_map = {
        'nested': LabeledArticleNested,
        'flat': LabeledArticleFlat,
        'unlabeled': UnlabeledArticle
    }

    schema = schema_map.get(format_type, UnlabeledArticle)

    # Validate
    valid, invalid = validate_articles(articles, schema)

    return valid, invalid, format_type
