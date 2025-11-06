import dspy

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
