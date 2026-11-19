"""Unit tests for preprocessing module using pytest."""

# pylint: disable=redefined-outer-name

import math

import pytest

from src import preprocessing as prep


@pytest.fixture
def sample_numeric_list():
    """Fixture providing a standard list of numbers for tests."""
    return [10.0, 20.0, 30.0, 40.0, 50.0]


# ==========================================
# CLEANING TESTS
# ==========================================


@pytest.mark.parametrize(
    "input_data, expected",
    [
        (["a", None, "", float("nan"), "b"], ["a", "b"]),
        ([1, 2, 3], [1, 2, 3]),
        ([None, ""], []),
        ([float("nan"), float("nan")], []),
        ([], []),
    ],
)
def test_remove_missing_values(input_data, expected):
    """Test removing None, empty strings, and NaN values."""
    assert prep.remove_missing_values(input_data) == expected


@pytest.mark.parametrize(
    "input_data, fill_val, expected",
    [
        ([1, None, 3], 0, [1, 0, 3]),
        (["a", "", "c"], "N/A", ["a", "N/A", "c"]),
        ([None, None], -1, [-1, -1]),
        ([1, 2, 3], 0, [1, 2, 3]),
        ([], 0, []),
    ],
)
def test_fill_missing_values(input_data, fill_val, expected):
    """Test replacing missing values with different fill values."""
    assert prep.fill_missing_values(input_data, fill_value=fill_val) == expected


def test_remove_duplicated_values():
    """Test duplicate removal for hashable and unhashable values."""
    data = ["apple", "banana", "apple", "orange", "banana"]

    assert prep.remove_duplicated_values(data) == [
        "apple",
        "banana",
        "orange",
    ]

    # Test the fallback implementation for unhashable values.
    unhashable_data = [[1, 2], [1, 2], [3]]

    assert prep.remove_duplicated_values(unhashable_data) == [
        [1, 2],
        [3],
    ]


def test_remove_duplicated_values_empty():
    """Test duplicate removal with an empty list."""
    assert not prep.remove_duplicated_values([])


# ==========================================
# NUMERICAL TESTS
# ==========================================


def test_normalize_min_max(sample_numeric_list):
    """Test min-max normalization with default and custom ranges."""
    result = prep.normalize_min_max(
        sample_numeric_list,
        new_min=0.0,
        new_max=1.0,
    )

    assert result == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize(
    "data, new_min, new_max, expected",
    [
        ([10.0, 20.0, 30.0], -1.0, 1.0, [-1.0, 0.0, 1.0]),
        ([-10.0, -5.0, 0.0, 5.0, 10.0], 0.0, 1.0, [0.0, 0.25, 0.5, 0.75, 1.0]),
        ([-10.0, 0.0, 10.0], -2.0, 2.0, [-2.0, 0.0, 2.0]),
        ([0.0, 5.0, 10.0], 1.0, -1.0, [1.0, 0.0, -1.0]),
    ],
)
def test_normalize_min_max_ranges(data, new_min, new_max, expected):
    """Test normalization with different input and target ranges."""
    result = prep.normalize_min_max(
        data,
        new_min=new_min,
        new_max=new_max,
    )

    assert result == pytest.approx(expected)


def test_normalize_min_max_empty_input():
    """Test min-max normalization with empty input."""
    assert not prep.normalize_min_max([])


def test_normalize_min_max_constant_values():
    """Test min-max normalization with constant values."""
    result = prep.normalize_min_max(
        [5.0, 5.0, 5.0],
        new_min=0.0,
        new_max=1.0,
    )

    assert result == [0.0, 0.0, 0.0]


def test_standardize_z_score(sample_numeric_list):
    """Test z-score standardization and edge cases."""
    result = prep.standardize_z_score(sample_numeric_list)

    expected = [
        -1.41421356237,
        -0.70710678118,
        0.0,
        0.70710678118,
        1.41421356237,
    ]

    assert result == pytest.approx(expected)

    # Verify that the resulting mean is approximately zero.
    mean = sum(result) / len(result)
    assert math.isclose(mean, 0.0, abs_tol=1e-7)

    # Test empty input.
    assert prep.standardize_z_score([]) == []

    # Test zero variance.
    assert prep.standardize_z_score([10.0, 10.0, 10.0]) == [
        0.0,
        0.0,
        0.0,
    ]

    # Test a single-element list.
    assert prep.standardize_z_score([5.0]) == [0.0]


def test_clip_values():
    """Test clipping values to specified boundaries."""
    data = [-5.0, 0.0, 0.5, 1.0, 2.5, 10.0]

    result = prep.clip_values(
        data,
        min_clip=0.0,
        max_clip=1.0,
    )

    assert result == [0.0, 0.0, 0.5, 1.0, 1.0, 1.0]


def test_convert_to_integers():
    """Test conversion of numeric strings and exclusion of invalid values."""
    data = ["10", "hello", "20.5", "abc", "30", "-5", "-2.8"]

    assert prep.convert_to_integers(data) == [10, 20, 30, -5, -2]

    # Test empty input.
    assert not prep.convert_to_integers([])

    # Test input containing no valid numbers.
    assert not prep.convert_to_integers(["hello", "abc"])


def test_log_transform():
    """Test logarithmic transformation of positive numbers."""
    data = [-1.0, 0.0, 1.0, math.e, math.e**2]

    result = prep.log_transform(data)

    expected = [0.0, 1.0, 2.0]

    assert result == pytest.approx(expected)

    # Test empty input.
    assert prep.log_transform([]) == []

    # Test input with no positive values.
    assert prep.log_transform([-5.0, 0.0, -1.0]) == []


def test_log_transform_single_positive_value():
    """Test logarithmic transformation with a single positive value."""
    assert prep.log_transform([math.e]) == pytest.approx([1.0])


# ==========================================
# TEXT TESTS
# ==========================================


def test_tokenize_text():
    """Test tokenization, punctuation removal, and lowercasing."""
    assert prep.tokenize_text("Hello World, MLOps 2026!") == "hello world mlops 2026"

    assert prep.tokenize_text("") == ""

    # Test punctuation-only input.
    assert prep.tokenize_text("!!!???") == ""


def test_select_alphanumeric_and_spaces():
    """Test selection of alphanumeric characters and spaces."""
    assert (
        prep.select_alphanumeric_and_spaces("Data-Science & MLOps!")
        == "DataScience  MLOps"
    )

    assert prep.select_alphanumeric_and_spaces("") == ""

    assert prep.select_alphanumeric_and_spaces("123 ABC!") == "123 ABC"


def test_remove_stopwords():
    """Test stopword removal and edge cases."""
    text = "This is a simple test"
    stopwords = ["this", "is", "a"]

    assert prep.remove_stopwords(text, stopwords) == "simple test"

    # Test empty text.
    assert prep.remove_stopwords("", stopwords) == ""

    # Test when all words are stopwords.
    assert prep.remove_stopwords("this is a", stopwords) == ""


# ==========================================
# STRUCTURAL TESTS
# ==========================================


def test_flatten_list():
    """Test flattening nested lists recursively."""
    nested = [[1, 2], [3, [4, 5]], 6]

    assert prep.flatten_list(nested) == [1, 2, 3, 4, 5, 6]

    # Test empty input.
    assert not prep.flatten_list([])

    # Test deeply nested input.
    assert prep.flatten_list([[[1]], [[2, [3]]]]) == [1, 2, 3]


def test_shuffle_list():
    """Test deterministic shuffling with a seed."""
    data = [1, 2, 3, 4, 5]

    shuffled_1 = prep.shuffle_list(data, seed=42)
    shuffled_2 = prep.shuffle_list(data, seed=42)

    assert shuffled_1 == shuffled_2
    assert sorted(shuffled_1) == sorted(data)


def test_shuffle_list_does_not_modify_original():
    """Test that shuffling does not modify the original list."""
    data = [1, 2, 3, 4, 5]
    original = data.copy()

    prep.shuffle_list(data, seed=42)

    assert data == original


def test_shuffle_list_empty():
    """Test shuffling an empty list."""
    assert not prep.shuffle_list([], seed=42)
