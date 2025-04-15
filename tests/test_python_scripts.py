"""Test suite for running Python scripts in a specific workflow order."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def load_workflow_config():
    """Load Python script workflow configuration from YAML file."""
    return {
        "workflow": {
            "execution_order": [
                {"script": "dist_xgboost/train.py", "args": []},
                {"script": "dist_xgboost/infer.py", "args": []},
                {"script": "dist_xgboost/serve.py", "args": []},
            ],
            "env": {},
            "timeout": 0,
            "stop_on_error": True,
        }
    }


def run_python_script(script_path, args=None, env=None, timeout=None):
    """Run a Python script with specified arguments and environment variables."""
    path = Path(script_path)
    assert path.exists(), f"Script not found: {script_path}"

    # Prepare command
    cmd = [sys.executable, str(path)]
    if args:
        cmd.extend(args)

    # Prepare environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    try:
        process = subprocess.run(
            cmd,
            env=run_env,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return True, process.stdout, process.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr
    except subprocess.TimeoutExpired:
        return False, "", f"Timeout of {timeout}s exceeded"


def test_python_script_workflow():
    """
    Run Python scripts in a specific order, preserving artifacts between executions.

    This test ensures that artifacts created by one script are accessible
    to subsequent scripts in the workflow.
    """
    # Load configuration
    config = load_workflow_config()
    script_execution_order = config["workflow"]["execution_order"]
    global_env = config["workflow"].get("env", {})
    timeout = config["workflow"].get("timeout", 0)
    timeout = None if timeout <= 0 else timeout
    stop_on_error = config["workflow"].get("stop_on_error", True)

    # Execute scripts in order
    for script_config in script_execution_order:
        # Handle both string format and dict format for backward compatibility
        if isinstance(script_config, str):
            script_path = script_config
            args = []
        else:
            script_path = script_config["script"]
            args = script_config.get("args", [])

        print(f"\nRunning script: {script_path} with args: {args}")

        # Run the script
        success, stdout, stderr = run_python_script(script_path, args=args, env=global_env, timeout=timeout)

        # Handle the result
        if success:
            print(f"Script {script_path} completed successfully")
            if stdout.strip():
                print(f"Output:\n{stdout}")
        else:
            print(f"Error in script {script_path}:")
            if stdout.strip():
                print(f"Standard output:\n{stdout}")
            if stderr.strip():
                print(f"Error output:\n{stderr}")

            if stop_on_error:
                pytest.fail(f"Script {script_path} failed with error: {stderr}")
            else:
                print("Continuing to next script due to stop_on_error=False")


if __name__ == "__main__":
    test_python_script_workflow()
