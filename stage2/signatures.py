"""
Stage 2 Signatures: DSPy signatures for flood verification and Ontario filtering
IMPROVED VERSION with clearer definitions and examples
"""
import dspy

class floodIdentification(dspy.Signature):
    """Determine if an article mentions a real flood event.
    
    Include any mention of actual flooding (past, present, or imminent), 
    even if brief or mentioned in passing. The goal is to identify all 
    articles that reference real flood events for historical research.
    
    Exclude only metaphorical usage or purely hypothetical discussions.
    """
    
    article_text: str = dspy.InputField(desc="Full text of the news article")
    title: str = dspy.InputField(desc="Article headline/title")
    
    flood_mentioned: bool = dspy.OutputField(
        desc="True if article mentions any real flood event, even briefly. False only for metaphorical or purely hypothetical."
    )
    reasoning: str = dspy.OutputField(
        desc="Explain what flood reference was found or why this is metaphorical/hypothetical."
    )


class isOntario(dspy.Signature):
    """Determine if a flood event occurred in Ontario, Canada.
    
    Look for Ontario cities, regions, rivers, or explicit mentions of "Ontario".
    Exclude other Canadian provinces and international locations.
    """
    
    article_text: str = dspy.InputField(desc="Full text of the flood article")
    title: str = dspy.InputField(desc="Article headline/title")
    
    is_ontario: bool = dspy.OutputField(
        desc="True if the flood occurred in Ontario, Canada. False otherwise."
    )
    reasoning: str = dspy.OutputField(
        desc="Explain what location indicators show this is/isn't in Ontario."
    )