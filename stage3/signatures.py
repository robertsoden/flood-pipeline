"""
DSPy signatures for Stage 3: Location and Date Extraction

Extracts structured information about flood location and date from verified flood articles.
"""
import dspy


class LocationExtraction(dspy.Signature):
    """
    Extract the location where a flood occurred from a newspaper article.

    The location should be as specific as possible (city, town, township, river, etc.)
    and should be in Ontario, Canada.
    """
    title: str = dspy.InputField(desc="Article title")
    article_text: str = dspy.InputField(desc="Full article text")

    location: str = dspy.OutputField(desc="Specific location where flood occurred (e.g., 'Thames River', 'Dover Township', 'Toronto')")
    reasoning: str = dspy.OutputField(desc="Brief explanation of how location was identified")


class DateExtraction(dspy.Signature):
    """
    Extract the date when a flood occurred from a newspaper article.

    The flood date is when the flood happened, which may be different from the
    article publication date. Extract month and year when available.
    """
    title: str = dspy.InputField(desc="Article title")
    article_text: str = dspy.InputField(desc="Full article text")
    publication_date: str = dspy.InputField(desc="Article publication date")

    flood_date: str = dspy.OutputField(desc="When flood occurred (e.g., 'March 1979', '1954', 'Spring 1960')")
    reasoning: str = dspy.OutputField(desc="Brief explanation of how date was determined")
