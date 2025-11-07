import dspy

# Define signatures
class floodIdentification(dspy.Signature):
    title: str = dspy.InputField(desc="headline/title of the newspaper article")
    article_text: str = dspy.InputField(desc="full text of newspaper article")
    flood_mentioned: bool = dspy.OutputField(desc="whether article mentions a specific historical flood event. "
             "Return True if the article mentions, references, or discusses an actual flood event that "
             "affected people, communities, or infrastructure - regardless of whether flooding is the main topic. "
             "This includes: (1) any mention of a real, specific flood (recent OR historical - from any time "
             "period within human history), (2) floods as context for other stories, (3) references to past floods "
             "in articles about other topics, (4) brief mentions or passing references to flood events. "
             "Return False for: metaphorical usage of 'flood' (e.g., 'flood of immigrants'), theoretical/hypothetical "
             "scenarios without a specific event, prehistoric/geological floods (ice age, ancient civilizations before "
             "written records), purely technical discussions of flood management with no specific event mentioned, "
             "or budget discussions that don't reference a specific flood event.")

class isOntario(dspy.Signature):
    title: str = dspy.InputField(desc="headline/title of the newspaper article")
    article_text: str = dspy.InputField(desc="full text of newspaper article")
    is_ontario: bool = dspy.OutputField(desc="whether the flood described occurred in Ontario, Canada")
