"""Integration tests for Click CLI commands using CliRunner."""

# pylint: disable=redefined-outer-name

import ast

import pytest
from click.testing import CliRunner

from src.cli import cli


@pytest.fixture
def runner():
    """Fixture providing a Click CLI test runner."""
    return CliRunner()


def extract_result(output, label):
    """Extract a Python value printed after a CLI output label."""
    content = output.split(label, 1)[1].strip()
    return ast.literal_eval(content)


# ==========================================
# CLEAN GROUP
# ==========================================


def test_cli_clean_remove_missing(runner):
    """Test integration of clean remove-missing command."""
    result = runner.invoke(
        cli,
        ["clean", "remove-missing", "apple", "", "10", "None"],
    )

    assert result.exit_code == 0
    cleaned = extract_result(result.output, "Cleaned data:")
    assert cleaned == ["apple", "10", "None"]


def test_cli_clean_fill_missing(runner):
    """Test integration of clean fill-missing command with options."""

    # Test integer parsing.
    result = runner.invoke(
        cli,
        ["clean", "fill-missing", "--fill-value", "42", "apple", "", "10"],
    )

    assert result.exit_code == 0
    filled = extract_result(result.output, "Filled data:")
    assert filled == ["apple", 42, "10"]

    # Test float parsing.
    result_float = runner.invoke(
        cli,
        ["clean", "fill-missing", "--fill-value", "3.14", "apple", "", "10"],
    )

    assert result_float.exit_code == 0
    filled_float = extract_result(result_float.output, "Filled data:")
    assert filled_float == ["apple", 3.14, "10"]

    # Test string parsing fallback.
    result_str = runner.invoke(
        cli,
        ["clean", "fill-missing", "--fill-value", "N/A", "apple", "", "10"],
    )

    assert result_str.exit_code == 0
    filled_str = extract_result(result_str.output, "Filled data:")
    assert filled_str == ["apple", "N/A", "10"]


# ==========================================
# NUMERIC GROUP
# ==========================================


def test_cli_numeric_normalize(runner):
    """Test integration of numeric normalize command with options."""
    result = runner.invoke(
        cli,
        [
            "numeric",
            "normalize",
            "10",
            "20",
            "30",
            "--new-min",
            "0",
            "--new-max",
            "1",
        ],
    )

    assert result.exit_code == 0
    normalized = extract_result(result.output, "Normalized data:")
    assert normalized == pytest.approx([0.0, 0.5, 1.0])


def test_cli_numeric_standardize(runner):
    """Test integration of numeric standardize command."""
    result = runner.invoke(
        cli,
        ["numeric", "standardize", "10", "20", "30", "40"],
    )

    assert result.exit_code == 0
    standardized = extract_result(result.output, "Standardized data:")

    expected = [
        -1.3416407865,
        -0.4472135955,
        0.4472135955,
        1.3416407865,
    ]

    assert standardized == pytest.approx(expected)


def test_cli_numeric_clip(runner):
    """Test integration of numeric clip command."""
    result = runner.invoke(
        cli,
        [
            "numeric",
            "clip",
            "--min-val",
            "0",
            "--max-val",
            "1",
            "--",
            "-5",
            "0.5",
            "2.5",
        ],
    )

    assert result.exit_code == 0
    clipped = extract_result(result.output, "Clipped data:")
    assert clipped == [0.0, 0.5, 1.0]


def test_cli_numeric_to_ints(runner):
    """Test integration of numeric to-ints command."""
    result = runner.invoke(
        cli,
        ["numeric", "to-ints", "10", "hello", "20.5", "30"],
    )

    assert result.exit_code == 0
    converted = extract_result(result.output, "Converted integers:")
    assert converted == [10, 20, 30]


def test_cli_numeric_log_scale(runner):
    """Test integration of numeric log-scale command."""
    result = runner.invoke(
        cli,
        ["numeric", "log-scale", "1", "10", "100"],
    )

    assert result.exit_code == 0
    transformed = extract_result(result.output, "Log transformed data:")

    assert transformed == pytest.approx([0.0, 2.30258509299, 4.60517018599])


# ==========================================
# TEXT GROUP
# ==========================================


def test_cli_text_tokenize(runner):
    """Test integration of text tokenize command."""
    result = runner.invoke(
        cli,
        ["text", "tokenize", "Hello MLOps 2026!"],
    )

    assert result.exit_code == 0
    assert "hello mlops 2026" in result.output


def test_cli_text_remove_punctuation(runner):
    """Test integration of text remove-punctuation command."""
    result = runner.invoke(
        cli,
        ["text", "remove-punctuation", "Data-Science & MLOps!"],
    )

    assert result.exit_code == 0
    assert "Processed text:" in result.output
    assert "DataScience  MLOps" in result.output


def test_cli_text_remove_stopwords(runner):
    """Test integration of text remove-stopwords command."""
    result = runner.invoke(
        cli,
        [
            "text",
            "remove-stopwords",
            "this is a simple test",
            "--stopwords",
            "this",
            "--stopwords",
            "is",
        ],
    )

    assert result.exit_code == 0
    assert "Text without stopwords:" in result.output
    assert "a simple test" in result.output


# ==========================================
# STRUCT GROUP
# ==========================================


def test_cli_struct_shuffle(runner):
    """Test integration of struct shuffle command with seed option."""
    result = runner.invoke(
        cli,
        ["struct", "shuffle", "1", "2", "3", "4", "5", "--seed", "42"],
    )

    assert result.exit_code == 0
    shuffled = extract_result(result.output, "Shuffled data:")

    assert shuffled == ["4", "2", "3", "5", "1"]


def test_cli_struct_flatten(runner):
    """Test integration of struct flatten command."""
    result = runner.invoke(
        cli,
        [
            "struct",
            "flatten",
            "[1, 2]",
            "[3, 4]",
            "unparsed_string",
        ],
    )

    assert result.exit_code == 0
    flattened = extract_result(result.output, "Flattened data:")

    assert flattened == [1, 2, 3, 4, "unparsed_string"]


def test_cli_struct_unique(runner):
    """Test integration of struct unique command."""
    result = runner.invoke(
        cli,
        [
            "struct",
            "unique",
            "apple",
            "banana",
            "apple",
            "orange",
        ],
    )

    assert result.exit_code == 0
    unique = extract_result(result.output, "Unique values:")

    assert unique == ["apple", "banana", "orange"]


# ==========================================
# MAIN / HELP GROUP
# ==========================================


def test_cli_main_help(runner):
    """Test main CLI group help output."""
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "Show this message and exit." in result.output
    assert "clean" in result.output
    assert "numeric" in result.output
    assert "text" in result.output
    assert "struct" in result.output


@pytest.mark.parametrize(
    "group",
    ["clean", "numeric", "text", "struct"],
)
def test_cli_group_help(runner, group):
    """Test help output for all CLI groups."""
    result = runner.invoke(cli, [group, "--help"])

    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "Show this message and exit." in result.output
