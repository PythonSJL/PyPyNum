from .types import real


def binomial(sample_size: int, successes: int, success_probability: real) -> float:
    from math import factorial
    if not isinstance(sample_size, int) or sample_size <= 0:
        raise ValueError("Sample size must be a positive integer")
    if not isinstance(successes, int) or successes < 0 or successes >= sample_size:
        raise ValueError("Number of successes must be a non-negative integer less than the sample size")
    if not isinstance(success_probability, (int, float)) or success_probability < 0 or success_probability > 1:
        raise ValueError("Success probability must be a number between 0 and 1")
    return factorial(sample_size) / (factorial(successes) * factorial(sample_size - successes)) * (
            success_probability ** successes) * ((1 - success_probability) ** (sample_size - successes))


def hypergeometric(total_items: int, success_items: int, sample_size: int, successes_in_sample: int) -> float:
    try:
        from math import comb
    except ImportError:
        from .maths import combination as comb
    if not isinstance(total_items, int) or total_items <= 0:
        raise ValueError("Total items must be a positive integer")
    if not isinstance(success_items, int) or success_items < 0 or success_items >= total_items:
        raise ValueError("Number of successful items must be a non-negative integer less than the total items")
    if not isinstance(sample_size, int) or sample_size <= 0 or sample_size >= total_items:
        raise ValueError("Sample size must be a positive integer less than the total items")
    if not isinstance(successes_in_sample, int) or successes_in_sample < 0 or successes_in_sample >= min(
            success_items, sample_size):
        raise ValueError("Number of successes in the sample must be a non-negative integer less than both the number of"
                         " successful items and the sample size")
    return comb(success_items, successes_in_sample) * comb(
        total_items - success_items, sample_size - successes_in_sample) / comb(total_items, sample_size)
