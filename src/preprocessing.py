"""Data preprocessing module for MLOps Lab 0.

Contains cleaning, numerical transformation, text, and structure functions.
"""

import math
import random
import re
from typing import Any, List, Union

# --- Internal Helper Functions ---


def _is_missing(item: Any) -> bool:
    """Checks if a given item represents a missing value (None, empty string, or NaN)."""
    if item is None or item == "":
        return True
    if isinstance(item, float) and math.isnan(item):
        return True
    return False


# --- 1. Data Cleaning ---


def remove_missing_values(data: List[Any]) -> List[Any]:
    """Removes missing values (such as None, empty strings, and NaN) from a list.

    Args:
        data: List of values containing potential missing elements.

    Returns:
        List of values with missing elements removed.
    """
    cleaned = []
    for item in data:
        if _is_missing(item):
            continue
        cleaned.append(item)
    return cleaned


def fill_missing_values(data: List[Any], fill_value: Any = 0) -> List[Any]:
    """Replaces missing values in a list with a specified fill value.

    Args:
        data: List of values containing potential missing elements.
        fill_value: The value used to replace missing elements (default is 0).

    Returns:
        List of values where missing elements are replaced with the fill value.
    """
    filled = []
    for item in data:
        if _is_missing(item):
            filled.append(fill_value)
        else:
            filled.append(item)
    return filled


def remove_duplicated_values(data: List[Any]) -> List[Any]:
    """Removes duplicate values from a list while preserving order.

    Args:
        data: List of values.

    Returns:
        List of unique values.
    """
    seen = set()
    unique_items = []
    for item in data:
        try:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        except TypeError:
            if item not in unique_items:
                unique_items.append(item)
    return unique_items


# --- 2. Numerical Attributes ---


def normalize_min_max(
    data: List[float], new_min: float = 0.0, new_max: float = 1.0
) -> List[float]:
    """Normalizes numerical values using the min-max scaling method.

    Args:
        data: List of numerical values.
        new_min: Lower bound of the new range (default to 0.0).
        new_max: Upper bound of the new range (default to 1.0).

    Returns:
        List of normalized numerical values.
    """
    if not data:
        return []

    min_val = min(data)
    max_val = max(data)

    if min_val == max_val:
        return [new_min for _ in data]

    normalized = []
    for x in data:
        norm_val = ((x - min_val) / (max_val - min_val)) * (new_max - new_min) + new_min
        normalized.append(norm_val)
    return normalized


def standardize_z_score(data: List[float]) -> List[float]:
    """Standardizes numerical values using the z-score method (mean=0, std=1).

    Args:
        data: List of numerical values.

    Returns:
        List of standardized numerical values.
    """
    if not data:
        return []

    n = len(data)
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    std_dev = math.sqrt(variance)

    if std_dev == 0.0:
        return [0.0 for _ in data]

    return [(x - mean) / std_dev for x in data]


def clip_values(data: List[float], min_clip: float, max_clip: float) -> List[float]:
    """Clips numerical values to a specific range.

    Args:
        data: List of numerical values.
        min_clip: Minimum threshold value.
        max_clip: Maximum threshold value.

    Returns:
        List of clipped numerical values.
    """
    clipped = []
    for x in data:
        if x < min_clip:
            clipped.append(min_clip)
        elif x > max_clip:
            clipped.append(max_clip)
        else:
            clipped.append(x)
    return clipped


def convert_to_integers(data: List[str]) -> List[int]:
    """Converts a list of strings to integers, excluding non-numerical values.

    Args:
        data: List of strings containing numerical and/or non-numerical values.

    Returns:
        List of values successfully converted to integers.
    """
    result = []
    for item in data:
        try:
            val = float(item)
            result.append(int(val))
        except (ValueError, TypeError):
            continue
    return result


def log_transform(data: List[float]) -> List[float]:
    """Applies a logarithmic scale transformation to positive numbers.

    Args:
        data: List of numerical values.

    Returns:
        List of values converted to logarithmic scale (restricted to original positives).
    """
    return [math.log(x) for x in data if x > 0]


# --- 3. Text Information ---


def select_alphanumeric_and_spaces(text: str) -> str:
    """Selects only alphanumeric characters and spaces from text.

    Args:
        text: Input string to be processed.

    Returns:
        Processed text string.
    """
    if not text:
        return ""
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def tokenize_text(text: str) -> str:
    """Tokenizes text into words, selecting alphanumeric characters and lowercasing.

    Args:
        text: Input string to be processed.

    Returns:
        Processed text string.
    """
    if not text:
        return ""
    cleaned = select_alphanumeric_and_spaces(text)
    return cleaned.lower()


def remove_stopwords(text: str, stop_words: List[str]) -> str:
    """Removes specified stop-words from a lower-cased text.

    Args:
        text: Input string to be processed.
        stop_words: List of words to be removed.

    Returns:
        Processed text string without stop-words.
    """
    if not text:
        return ""

    lowered_text = text.lower()
    words = lowered_text.split()
    stop_words_set = {sw.lower() for sw in stop_words}

    filtered_words = [word for word in words if word not in stop_words_set]
    return " ".join(filtered_words)


# --- 4. Data Structure ---


def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flattens a list of lists into a single-level list.

    Args:
        nested_list: A list containing inner lists.

    Returns:
        A single flat list containing all elements.
    """
    flat = []
    for sublist in nested_list:
        if isinstance(sublist, list):
            flat.extend(flatten_list(sublist))
        else:
            flat.append(sublist)
    return flat


def shuffle_list(data: List[Any], seed: Union[int, None] = None) -> List[Any]:
    """Randomly shuffles a list of values with an optional seed for reproducibility.

    Args:
        data: List of values to be shuffled.
        seed: Random seed for reproducibility (default to None).

    Returns:
        A new list with elements randomly shuffled.
    """
    shuffled_data = list(data)
    if seed is not None:
        random.seed(seed)
    random.shuffle(shuffled_data)
    return shuffled_data
