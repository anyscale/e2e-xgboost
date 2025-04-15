"""Test suite for validating Jupyter notebooks in the repository."""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import jupytext
import pytest


def filter_out_cells_with_remove_cell_ci_tag(cells: list):
    """Filters out cells which contain the 'remove-cell-ci' tag in metadata"""

    def should_keep_cell(cell):
        tags = cell.metadata.get("tags")
        if tags:
            # Both - and _ for consistent behavior with built-in tags
            return "remove_cell_ci" not in tags and "remove-cell-ci" not in tags
        return True

    return [cell for cell in cells if should_keep_cell(cell)]


def postprocess_notebook(notebook):
    """Process notebook by removing cells marked with remove-cell-ci tag."""
    notebook.cells = filter_out_cells_with_remove_cell_ci_tag(notebook.cells)
    return notebook


def run_notebook(notebook_path, postprocess=True):
    """
    Convert a notebook to a Python script and execute it.

    Args:
        notebook_path: Path to the notebook file
        postprocess: If True, filter out cells with remove-cell-ci tag
    """
    path = Path(notebook_path)
    assert path.exists(), f"Notebook not found: {notebook_path}"

    with open(path, "r") as f:
        notebook = jupytext.read(f)

    if postprocess:
        notebook = postprocess_notebook(notebook)

    # Define the display function available in notebooks
    display_function = """
def display(*args, **kwargs):
    print(*args, **kwargs)
"""

    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        # Add display function
        f.write(display_function)
        # Convert notebook to Python script
        jupytext.write(notebook, f, fmt="py:percent")
        script_path = f.name

    try:
        # Run the notebook script
        subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"Error output: {e.stderr}")
        print(f"Standard output: {e.stdout}")
        raise
    finally:
        # Clean up temp file
        if os.path.exists(script_path):
            os.unlink(script_path)


def get_notebooks():
    """Find all notebooks in the repository."""
    notebooks_dir = Path("notebooks")
    return list(notebooks_dir.glob("*.ipynb"))


def load_workflow_config():
    """Load notebook workflow configuration from YAML file."""
    return {
        "workflow": {
            "execution_order": [
                "notebooks/01-Distributed_Training.ipynb",
                "notebooks/02-Validation.ipynb",
                "notebooks/03-Serving.ipynb",
            ],
            "timeout": 0,
            "stop_on_error": True,
        }
    }


# Load execution order from config
config = load_workflow_config()
NOTEBOOK_EXECUTION_ORDER = config["workflow"]["execution_order"]
TIMEOUT = config["workflow"].get("timeout", 0)
STOP_ON_ERROR = config["workflow"].get("stop_on_error", True)


@pytest.mark.parametrize("notebook_path", get_notebooks())
def test_notebook(notebook_path):
    """Test that each notebook runs without errors."""
    run_notebook(notebook_path)


def test_notebook_workflow():
    """
    Run all notebooks in a specific order, preserving artifacts between executions.

    This test ensures that artifacts created by one notebook are accessible
    to subsequent notebooks in the workflow.
    """
    # Set up working directory for workflow
    workflow_dir = os.getcwd()

    for notebook_path in NOTEBOOK_EXECUTION_ORDER:
        print(f"\nRunning notebook: {notebook_path}")
        path = Path(notebook_path)
        assert path.exists(), f"Notebook not found: {notebook_path}"

        with open(path, "r") as f:
            notebook = jupytext.read(f)

        notebook = postprocess_notebook(notebook)

        # Define the display function available in notebooks
        display_function = """
def display(*args, **kwargs):
    print(*args, **kwargs)
"""

        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(display_function)
            jupytext.write(notebook, f, fmt="py:percent")
            script_path = f.name

        try:
            # Run the notebook script in the workflow directory to preserve artifacts
            timeout = None if TIMEOUT <= 0 else TIMEOUT
            subprocess.run(
                [sys.executable, script_path],
                check=True,
                cwd=workflow_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            print(f"Notebook {notebook_path} completed successfully")

        except subprocess.CalledProcessError as e:
            print(f"Error in notebook {notebook_path}:")
            print(f"Error output: {e.stderr}")
            print(f"Standard output: {e.stdout}")
            if STOP_ON_ERROR:
                raise
            print("Continuing to next notebook due to stop_on_error=False")

        except subprocess.TimeoutExpired:
            print(f"Timeout of {TIMEOUT}s exceeded for notebook {notebook_path}")
            if STOP_ON_ERROR:
                raise
            print("Continuing to next notebook due to stop_on_error=False")

        finally:
            if os.path.exists(script_path):
                os.unlink(script_path)
