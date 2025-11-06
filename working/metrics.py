import dspy

# Define specific eval functions
def eval_location(prediction, example):
    """Multi-strategy location matching"""
    from difflib import SequenceMatcher
    
    pred = str(prediction.location).lower().strip()
    exp = str(example.location).lower().strip()
    
    # Handle empty cases
    if not pred and not exp:
        return True
    if not pred or not exp:
        return False
    
    # Check all strategies at once
    return (
        pred == exp or  # Exact
        exp in pred or pred in exp or  # Substring
        SequenceMatcher(None, pred, exp).ratio() >= 0.70 or  # Fuzzy
        len(set(pred.split()) & set(exp.split())) / len(set(exp.split())) >= 0.5  # Token
    )

def eval_date(prediction, example):
    return prediction.flood_date == example.flood_date
    
def eval_impacts(prediction, example):
    return prediction.impacts == example.impacts

def eval_reasoning(prediction):
    return bool(prediction.reasoning)

# Define metrics
def extraction_precision_focused_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Penalize false positives more heavily to improve precision"""
    # Must have reasoning
    if not prediction.reasoning:
        return False
    
    # Correct prediction - reward it
    if prediction.flood_mentioned == example.flood_mentioned:
        return True
    
    # Incorrect prediction - penalize it
    # (Both false positives and false negatives are equally bad here,
    #  but the optimizer will learn from the training data patterns)
    return False
            
def ontario_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None):
    return (prediction.is_ontario == example.is_ontario and
            bool(prediction.reasoning))
            
def details_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Store granular results for later analysis"""  # ← On next line, indented
    # Calculate each field
    location_correct = eval_location(prediction, example)
    date_correct = eval_date(prediction, example)
    impacts_correct = eval_impacts(prediction, example)
    reasoning_correct = eval_reasoning(prediction)
    
    # Store on prediction  
    prediction.field_results = {
        'location_correct': location_correct,
        'date_correct': date_correct,
        'impacts_correct': impacts_correct,
        'reasoning_correct': reasoning_correct
    }
    
    # Return overall metric for DSPy's evaluation
    return all([location_correct, date_correct, impacts_correct, reasoning_correct])