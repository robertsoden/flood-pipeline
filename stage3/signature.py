"""
Stage 2 Signatures: DSPy signatures for flood verification and Ontario filtering
IMPROVED VERSION with clearer definitions and examples
"""
import dspy


class floodIdentification(dspy.Signature):
    """Determine if an article describes an ACTUAL flood event.
    
    A flood is water overflowing onto normally dry land, causing damage or disruption.
    
    ✅ FLOODS (include these):
    - Rivers, lakes, or streams overflowing their banks
    - Heavy rainfall causing street/basement flooding
    - Storm surge, coastal flooding, tidal flooding
    - Dam failures, levee breaks
    - Flash floods, urban flooding
    - Snowmelt flooding
    - Ice jam flooding
    - Actual ongoing or recent flood events
    
    ❌ NOT FLOODS (exclude these):
    - Metaphorical "flood" (flood of emails, flood of migrants, flood of applications)
    - Financial/market flooding (flooding the market with products)
    - Future flood risk, flood planning, flood insurance discussions WITHOUT current event
    - Historical floods mentioned in passing (e.g., "since the 1950 flood")
    - Flood warnings without actual flooding occurring
    - Flood zone designations, flood maps
    - Construction projects to prevent future floods
    - Biblical/religious flood references
    - Brand names containing "flood" (e.g., "Flood Insurance Company")
    
    ⚠️ EDGE CASES:
    - "Flood warning" WITH flooding happening = FLOOD
    - "Flood warning" with NO flooding yet = NOT FLOOD
    - Historical flood WITH current anniversary/commemoration event = DEPENDS (if current event, yes; if just mention, no)
    
    Focus on: Is there CURRENTLY water where it shouldn't be, causing problems?
    """
    
    article_text: str = dspy.InputField(desc="Full text of the news article")
    title: str = dspy.InputField(desc="Article headline/title")
    
    flood_mentioned: bool = dspy.OutputField(
        desc="True if article describes an actual flood event currently happening or recently happened. False for metaphorical use, future planning, or historical mentions."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation: What specific evidence indicates this is/isn't a flood? Quote key phrases from the article."
    )


class isOntario(dspy.Signature):
    """Determine if a flood event occurred in Ontario, Canada.
    
    Ontario indicators:
    - Cities: Toronto, Ottawa, Hamilton, London, Windsor, Kitchener, Mississauga, Brampton, etc.
    - Regions: Greater Toronto Area (GTA), Niagara Region, Muskoka, Northern Ontario, etc.
    - Rivers: Grand River, Ottawa River, Thames River, Credit River, etc.
    - Lakes: Lake Ontario, Lake Erie, Lake Huron (Ontario side)
    - Highways: Highway 401, QEW, Highway 400, etc.
    - Provinces mentioned: "Ontario" explicitly
    - Canadian context: mentions "Canada" or "Canadian" with Ontario location
    
    NOT Ontario:
    - Other Canadian provinces: Quebec, British Columbia, Alberta, Manitoba, etc.
    - US locations: even if near Ontario border (e.g., Buffalo, Detroit, Cleveland)
    - International locations
    - Generic "Canada" without specific Ontario location
    
    Edge cases:
    - Ottawa River: borders Ontario and Quebec - include if Ontario side mentioned
    - Niagara: could be Niagara Falls, NY (US) or Niagara, ON (Canada) - check context
    - London: could be London, UK or London, Ontario - check context for Canada
    - Border cities: Check which side of border (e.g., Windsor, ON vs Detroit, MI)
    """
    
    article_text: str = dspy.InputField(desc="Full text of the flood article")
    title: str = dspy.InputField(desc="Article headline/title")
    
    is_ontario: bool = dspy.OutputField(
        desc="True if the flood occurred in Ontario, Canada. False for other locations including other Canadian provinces and US locations."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation: What specific location indicators show this is/isn't in Ontario? Quote location names from the article."
    )