"""Command Line Interface (CLI) for MLOps Lab 0 using Click."""

import json
import click
from src import preprocessing as prep


@click.group()
def cli():
    """Main CLI group for MLOps Lab 0 data preprocessing operations."""


# ==========================================
# 1. CLEAN GROUP
# ==========================================
@cli.group()
def clean():
    """Commands related to data cleaning (removing and filling missing values)."""


@clean.command(name="remove-missing")
@click.argument("data", nargs=-1)
def remove_missing_cmd(data):
    """Removes missing values (None, '', NaN) from a list of values.

    Example:
        uv run python -m src.cli clean remove-missing apple "" 10 None
    """
    cleaned = prep.remove_missing_values(list(data))
    click.echo(f"Cleaned data: {cleaned}")


@clean.command(name="fill-missing")
@click.option(
    "--fill-value",
    default="0",
    help="Value used to replace missing elements (default is 0).",
)
@click.argument("data", nargs=-1)
def fill_missing_cmd(fill_value, data):
    """Replaces missing values in a list with a specified fill value.

    Example:
        uv run python -m src.cli clean fill-missing --fill-value N/A apple "" 10
    """
    try:
        if "." in fill_value:
            parsed_fill = float(fill_value)
        else:
            parsed_fill = int(fill_value)
    except ValueError:
        parsed_fill = fill_value

    filled = prep.fill_missing_values(list(data), fill_value=parsed_fill)
    click.echo(f"Filled data: {filled}")


# ==========================================
# 2. NUMERIC GROUP
# ==========================================
@cli.group()
def numeric():
    """Commands related to numerical attributes transformations."""


@numeric.command(name="normalize")
@click.option(
    "--new-min", type=float, default=0.0, help="New minimum value (default: 0)."
)
@click.option(
    "--new-max", type=float, default=1.0, help="New maximum value (default: 1)."
)
@click.argument("data", type=float, nargs=-1)
def normalize_cmd(new_min, new_max, data):
    """Normalizes numerical values using the min-max method.

    Example:
        uv run python -m src.cli numeric normalize 10 20 30 --new-min 0 --new-max 1
    """
    normalized = prep.normalize_min_max(list(data), new_min=new_min, new_max=new_max)
    click.echo(f"Normalized data: {normalized}")


@numeric.command(name="standardize")
@click.argument("data", type=float, nargs=-1)
def standardize_cmd(data):
    """Standardizes numerical values using the z-score method.

    Example:
        uv run python -m src.cli numeric standardize 10 20 30 40
    """
    standardized = prep.standardize_z_score(list(data))
    click.echo(f"Standardized data: {standardized}")


@numeric.command(name="clip")
@click.option(
    "--min-val", type=float, default=0.0, help="Minimum value to clip (default: 0)."
)
@click.option(
    "--max-val", type=float, default=1.0, help="Maximum value to clip (default: 1)."
)
@click.argument("data", type=float, nargs=-1)
def clip_cmd(min_val, max_val, data):
    """Clips numerical values to a certain range.

    Example:
        uv run python -m src.cli numeric clip -5 0.5 2.5 --min-val 0 --max-val 1
    """
    clipped = prep.clip_values(list(data), min_clip=min_val, max_clip=max_val)
    click.echo(f"Clipped data: {clipped}")


@numeric.command(name="to-ints")
@click.argument("data", nargs=-1)
def to_ints_cmd(data):
    """Converts a list of strings to integers, excluding non-numerical values.

    Example:
        uv run python -m src.cli numeric to-ints 10 "hello" 20.5 30
    """
    converted = prep.convert_to_integers(list(data))
    click.echo(f"Converted integers: {converted}")


@numeric.command(name="log-scale")
@click.argument("data", type=float, nargs=-1)
def log_scale_cmd(data):
    """Applies logarithmic scale transformation to positive numbers.

    Example:
        uv run python -m src.cli numeric log-scale 1 10 100
    """
    transformed = prep.log_transform(list(data))
    click.echo(f"Log transformed data: {transformed}")


# ==========================================
# 3. TEXT GROUP
# ==========================================
@cli.group()
def text():
    """Commands to deal with textual information."""


@text.command(name="tokenize")
@click.argument("text_input")
def tokenize_cmd(text_input):
    """Tokenizes text into words, selecting alphanumeric and lowercasing.

    Example:
        uv run python -m src.cli text tokenize "Hello World, MLOps 2026!"
    """
    tokenized = prep.tokenize_text(text_input)
    click.echo(f"Tokenized text: {tokenized}")


@text.command(name="remove-punctuation")
@click.argument("text_input")
def remove_punctuation_cmd(text_input):
    """Selects only alphanumeric characters and spaces from text.

    Example:
        uv run python -m src.cli text remove-punctuation "Data-Science & MLOps!"
    """
    processed = prep.select_alphanumeric_and_spaces(text_input)
    click.echo(f"Processed text: {processed}")


@text.command(name="remove-stopwords")
@click.option(
    "--stopwords",
    multiple=True,
    help=("Help list of stop words to remove " "(can be passed multiple times)."),
)
@click.argument("text_input")
def remove_stopwords_cmd(stopwords, text_input):
    """Removes specified stop-words from a text.

    Example:
        uv run python -m src.cli text remove-stopwords \
            "this is a simple test" --stopwords this --stopwords is
    """
    stop_list = list(stopwords) if stopwords else ["the", "is", "at", "which", "and"]
    cleaned_text = prep.remove_stopwords(text_input, stop_words=stop_list)
    click.echo(f"Text without stopwords: {cleaned_text}")


# ==========================================
# 4. STRUCT GROUP
# ==========================================
@cli.group()
def struct():
    """Commands related to the structure of data (shuffle, flatten, unique)."""


@struct.command(name="shuffle")
@click.option(
    "--seed", type=int, default=None, help="Integer seed for reproducibility."
)
@click.argument("data", nargs=-1)
def shuffle_cmd(seed, data):
    """Randomly shuffles a list of values.

    Example:
        uv run python -m src.cli struct shuffle 1 2 3 4 5 --seed 42
    """
    shuffled = prep.shuffle_list(list(data), seed=seed)
    click.echo(f"Shuffled data: {shuffled}")


@struct.command(name="flatten")
@click.argument("data", nargs=-1)
def flatten_cmd(data):
    """Flattens a list of inputs.

    Example:
        uv run python -m src.cli struct flatten "[1, 2]" "[3, 4]"
    """
    parsed_data = []
    for item in data:
        try:
            parsed_item = json.loads(item)
            if isinstance(parsed_item, list):
                parsed_data.append(parsed_item)
            else:
                parsed_data.append(item)
        except (json.JSONDecodeError, TypeError):
            parsed_data.append(item)

    flattened = prep.flatten_list(parsed_data)
    click.echo(f"Flattened data: {flattened}")


@struct.command(name="unique")
@click.argument("data", nargs=-1)
def unique_cmd(data):
    """Removes duplicate values from a list while preserving order.

    Example:
        uv run python -m src.cli struct unique apple banana apple orange
    """
    unique_vals = prep.remove_duplicated_values(list(data))
    click.echo(f"Unique values: {unique_vals}")


if __name__ == "__main__":
    cli()  # pragma: no cover
