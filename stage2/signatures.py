"""
Stage 2 Signatures: DSPy signatures for flood verification and Ontario filtering

These signatures determine which articles pass to Stage 3 for extraction.
The goal is to capture all articles that could provide evidence of historical
Ontario floods, including brief mentions in obituaries, anniversary articles, etc.
"""
import dspy


class floodIdentification(dspy.Signature):
    """Determine if an article describes or references a real flood event.

    INCLUDE articles that:
    - Describe a flood event (past, present, or imminent)
    - Mention a historical flood in context (e.g., "survived the 1954 flood")
    - Reference flood damage, flood victims, or flood recovery
    - Discuss flood prevention/mitigation related to a specific event

    EXCLUDE articles that:
    - Use "flood" only metaphorically ("flood of applications", "flooded with calls")
    - Discuss flooding only hypothetically without reference to real events
    - Are about non-water floods (flood lights, flood insurance policies without events)
    - Only mention flooding in a completely different country with no Ontario connection
    """

    article_text: str = dspy.InputField(desc="Full text of the news article")
    title: str = dspy.InputField(desc="Article headline/title")

    flood_mentioned: bool = dspy.OutputField(
        desc="True if article references any real flood event. False for metaphorical, hypothetical, or non-water 'flood' usage."
    )
    reasoning: str = dspy.OutputField(
        desc="Explain what flood event was referenced, or why this is metaphorical/not a real flood."
    )


class isOntario(dspy.Signature):
    """Determine if the flood event described occurred in Ontario, Canada.

    The article must describe a flood that OCCURRED IN Ontario, not just:
    - A flood mentioned by an Ontario newspaper but located elsewhere
    - A flood affecting Ontarians who were traveling elsewhere
    - A flood in another province (Manitoba, Quebec, etc.) or country

    INCLUDE if:
    - Flood location is an Ontario city, town, or region
    - Flood is on an Ontario river, lake, or watershed
    - Article explicitly states the flood was in Ontario

    EXCLUDE if:
    - Flood occurred in another Canadian province (even if article is from Ontario)
    - Flood occurred in another country
    - No specific location is mentioned for the flood
    - The only Ontario connection is the newspaper's location, not the flood's location
    """

    article_text: str = dspy.InputField(desc="Full text of the flood article")
    title: str = dspy.InputField(desc="Article headline/title")

    is_ontario: bool = dspy.OutputField(
        desc="True ONLY if the flood occurred in Ontario, Canada. False if flood was elsewhere or location unclear."
    )
    reasoning: str = dspy.OutputField(
        desc="Identify the flood location and explain why it is/isn't in Ontario."
    )
