import dspy

# Load data and prep for DSPy
def prepare_data(raw_data: list) -> list: 
    examples = []
    
    for item in raw_data:
        # Check if data has nested annotations format (original)
        if 'annotations' in item:
            # Original format with nested annotations
            annotations = item['annotations']
            article_text = item.get('full_text', '')
            publication_date = item.get('date', '')
            flood_mentioned = annotations.get('flood_mentioned', False)
            location = annotations.get('location', '')
            flood_date = annotations.get('flood_date', '')
            impacts = annotations.get('impacts', '')
            is_ontario = annotations.get('is_ontario', False)
        else:
            # Flat format from combined data
            article_text = item.get('article_text', item.get('full_text', ''))
            publication_date = item.get('publication_date', item.get('date', ''))
            flood_mentioned = item.get('flood_mentioned', False)
            location = item.get('location', '')
            flood_date = item.get('flood_date', '')
            impacts = item.get('impacts', '')
            is_ontario = item.get('is_ontario', False)
        
        # Create a DSPy Example 
        example = dspy.Example(
            article_text=article_text,
            publication_date=publication_date,
            flood_mentioned=flood_mentioned,
            location=location,
            flood_date=flood_date,
            impacts=impacts,
            is_ontario=is_ontario
        ).with_inputs('article_text', 'publication_date')
        
        examples.append(example)
    
    return examples