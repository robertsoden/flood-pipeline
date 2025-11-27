"""
Stage 3 Signatures: Extract flood date and location from articles
"""
import dspy

class FloodLocationExtraction(dspy.Signature):
    """Extract the location and date of a flood event from a news article.

    Location should be a specific place in Ontario (city, town, region, or river).
    Date should be when the flood occurred (not publication date).
    If date is not explicitly stated, indicate this clearly.
    """

    article_text: str = dspy.InputField(desc="Full text of the flood article")
    title: str = dspy.InputField(desc="Article headline/title")

    location: str = dspy.OutputField(
        desc="The primary location where the flood occurred (city, town, region, or river in Ontario). Be specific."
    )
    flood_date: str = dspy.OutputField(
        desc="When the flood occurred (YYYY-MM-DD format if possible, or description like 'early May 1997' if exact date unknown). If not mentioned, say 'date not specified'."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of how you identified the location and date from the article."
    )


class FloodVerification(dspy.Signature):
    """Verify and improve flood location and date extraction from a news article.

    You are given:
    1. A news article about flooding in Ontario
    2. The article's PUBLICATION DATE (when the article was written/published)
    3. A suggested location extracted by NER (may be correct, incorrect, or "not found")

    Your task:
    1. VERIFY the suggested location - is this actually where the flood occurred?
       - The location should be in Ontario, Canada
       - It should be the PRIMARY flood location, not just a place mentioned
       - If suggested location is wrong or "not found", extract the correct one

    2. EXTRACT the flood date - when did the flood actually occur?
       - This is NOT the publication date (which is provided separately)
       - News articles are typically published 0-3 days AFTER a flood event
       - Resolve relative date mentions using the publication date:
         * "yesterday" = 1 day before publication date
         * "last week" = approximately 7 days before publication date
         * "earlier this month" = earlier in the same month as publication
         * "two days ago" = 2 days before publication date
       - If the article discusses an ongoing flood, the date is likely close to publication
       - If no date is mentioned, estimate based on context (e.g., "early [publication month]")
    """

    article_text: str = dspy.InputField(desc="Full text of the flood article")
    title: str = dspy.InputField(desc="Article headline/title")
    publication_date: str = dspy.InputField(
        desc="When this article was published (e.g., 'Aug 9, 2001'). Use this to resolve relative dates like 'yesterday' or 'last week'."
    )
    suggested_location: str = dspy.InputField(
        desc="Location extracted by NER system (may be 'not found' or incorrect). Verify this is the actual flood location."
    )

    location: str = dspy.OutputField(
        desc="The verified/corrected primary flood location in Ontario (city, town, region, or river). Be specific. Use 'not found' ONLY if no Ontario flood location exists in the article."
    )
    location_verified: bool = dspy.OutputField(
        desc="True if the suggested_location was correct, False if you corrected it or found a new one."
    )
    flood_date: str = dspy.OutputField(
        desc="STRICT FORMAT: Must be 'Month YYYY' (e.g., 'June 1950', 'March 2003'). For decades use 'YYYY' or '1950s'. For ranges within a year use 'Month-Month YYYY'. Use 'not found' ONLY if no flood date can be determined. NEVER use 'not applicable' or other phrases - only 'Month YYYY' format or 'not found'."
    )
    date_confidence: str = dspy.OutputField(
        desc="How confident in the date: 'high' (explicit date mentioned), 'medium' (relative date resolved), 'low' (estimated from context). Use 'none' if flood_date is 'not found'."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of verification and any date calculations (e.g., 'yesterday from Aug 9, 2001 = Aug 8, 2001')."
    )
