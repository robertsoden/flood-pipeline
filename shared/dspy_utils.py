"""
DSPy configuration utilities.

Provides centralized LLM configuration to avoid code duplication across stages.
"""
import os
import dspy
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


def configure_dspy_lm(
    model_config: Dict[str, Any],
    temperature: float = 0.1,
    mode: str = 'inference'
) -> dspy.LM:
    """
    Configure DSPy language model with proper settings.

    Handles API key detection, rate limiting, and model selection.

    Args:
        model_config: MODEL_CONFIG dictionary from shared.config
        temperature: Temperature for LLM calls
        mode: 'inference' (deterministic) or 'optimization' (exploratory)

    Returns:
        Configured dspy.LM instance

    Example:
        >>> from shared.config import MODEL_CONFIG, get_temperature, STAGE2_CONFIG
        >>> temp = get_temperature(STAGE2_CONFIG, mode='inference')
        >>> lm = configure_dspy_lm(MODEL_CONFIG, temperature=temp)
    """
    # Check for Anthropic API key
    anthropic_key = os.getenv('ANTHROPIC_API_KEY')

    # Build LM kwargs
    lm_kwargs = {'temperature': temperature}

    if anthropic_key:
        # Use configured Anthropic model
        model_name = model_config.get('name', 'anthropic/claude-sonnet-4-20250514')
        lm_kwargs['api_key'] = anthropic_key

        # Configure retries for rate limiting
        try:
            import litellm
            litellm.num_retries = 5
            litellm.request_timeout = 120
        except ImportError:
            pass

        logger.info(f"Using Anthropic model: {model_name}")

    else:
        # Fall back to configured model (Ollama, Bedrock, etc.)
        model_name = model_config.get('name', 'ollama_chat/qwen2.5:14b-instruct-q5_K_M')

        if model_config.get('api_base'):
            lm_kwargs['api_base'] = model_config['api_base']
        if model_config.get('api_key'):
            lm_kwargs['api_key'] = model_config['api_key']

        logger.info(f"Using configured model: {model_name}")

    # Create and configure LM
    lm = dspy.LM(model_name, **lm_kwargs)
    dspy.configure(lm=lm)

    logger.info(f"DSPy configured: {model_name} (temperature={temperature}, mode={mode})")

    return lm


def get_configured_lm() -> Optional[dspy.LM]:
    """
    Get the currently configured DSPy LM.

    Returns:
        Current LM or None if not configured
    """
    try:
        return dspy.settings.lm
    except Exception:
        return None


def load_optimized_model(
    model_path: str,
    signature_class: type,
    predictor_class: type = None
) -> Any:
    """
    Load an optimized DSPy model from file.

    Args:
        model_path: Path to saved model JSON file
        signature_class: DSPy Signature class for the model
        predictor_class: Predictor class (default: ChainOfThought)

    Returns:
        Loaded predictor instance

    Example:
        >>> from stage2.signatures import floodIdentification
        >>> model = load_optimized_model(
        ...     'models/stage2_flood_verified.json',
        ...     floodIdentification
        ... )
    """
    if predictor_class is None:
        predictor_class = dspy.ChainOfThought

    predictor = predictor_class(signature_class)
    predictor.load(str(model_path))

    logger.info(f"Loaded optimized model from: {model_path}")

    return predictor


def create_evaluator(
    devset: list,
    metric: callable,
    num_threads: int = 2,
    max_errors: int = 10,
    display_progress: bool = True
) -> dspy.Evaluate:
    """
    Create a configured DSPy Evaluate instance with error tolerance.

    Args:
        devset: Evaluation dataset
        metric: Metric function
        num_threads: Number of parallel threads
        max_errors: Maximum errors to tolerate before failing
        display_progress: Show progress bar

    Returns:
        Configured Evaluate instance
    """
    return dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=True,
        max_errors=max_errors,
        failure_score=0.0
    )
