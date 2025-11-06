import dspy

# Define signatures
class floodIdentification(dspy.Signature):
    article_text: str = dspy.InputField(desc="full text of newspaper article")
    flood_mentioned: bool = dspy.OutputField(desc="whether article is primarily about a recent/current specific flood event. "
             "Return True only if: (1) the flood is a main focus of the article, not just "
             "a brief mention or budget line item, (2) the flood is recent (within a few years "
             "of publication date), not ancient history or geological events, and (3) it's an "
             "actual flood, not metaphorical usage. Return False for: passing references to past "
             "floods, budget mentions of flood programs, ancient/prehistoric floods, tsunamis "
             "mentioned only briefly, or articles where flood is tangential context.")

class isOntario(dspy.Signature):
    article_text: str = dspy.InputField(desc="full text of newspaper article")
    is_ontario: bool = dspy.OutputField(desc="whether the flood described occurred in Ontario, Canada")
