# Lab 0: Fundamentals of Continuous Integration

A data preprocessing library and Command Line Interface (CLI) built with Python and `Click`, managed with `uv`, and verified using `pytest`, `black`, and `pylint`.

---

## Setup

Ensure you have [`uv`](https://docs.astral.sh/uv/) installed, then set up the environment and dependencies:

```bash
uv sync
```

---

## Project Structure

```text
Lab0/
├── pyproject.toml       # Project configuration and dependencies
├── src/
│   ├── preprocessing.py # Core preprocessing functions (clean, numeric, text, struct)
│   └── cli.py           # Click CLI implementation
└── tests/
    ├── test_logic.py    # Unit tests with pytest
    └── test_cli.py      # Integration tests with CliRunner
```

---

## CLI Usage

Run the CLI using `uv run python -m src.cli` followed by the group and command:

```bash
# Data Cleaning
uv run python -m src.cli clean remove-missing apple "" 10
uv run python -m src.cli clean fill-missing --fill-value 0 1 None 3

# Numerical Transformations
uv run python -m src.cli numeric normalize 10 20 30 --new-min 0 --new-max 1
uv run python -m src.cli numeric standardize 10 20 30 40
uv run python -m src.cli numeric clip --min-val 0 --max-val 1 -- -5 0.5 2.5

# Text Processing
uv run python -m src.cli text tokenize "Hello World, MLOps 2026!"
uv run python -m src.cli text remove-stopwords "this is a simple test" --stopwords this --stopwords is

# Data Structures
uv run python -m src.cli struct shuffle 1 2 3 4 5 --seed 42
uv run python -m src.cli struct flatten "[1, 2]" "[3, 4]"
uv run python -m src.cli struct unique apple banana apple orange
```

You can view available options for any command with `--help`:

```bash
uv run python -m src.cli --help
```

---

## Testing & Quality Assurance

Run the test suite and quality tools with the following commands:

- **Run all tests:**
  ```bash
  uv run python -m pytest -v
  ```

- **Run tests with coverage report:**
  ```bash
  uv run python -m pytest -v --cov=src
  ```

- **Check code formatting with Black:**
  ```bash
  uv run black --check src/*.py
  ```

- **Run linting with Pylint:**
  ```bash
  uv run python -m pylint src/*.py
  ```

