# Continuous Integration & Kaggle Execution Harness

## Overview
Automated execution harness and script runner for training and evaluating smart-grouping workloads on Kaggle GPU environments triggered via GitHub Actions workflows.

## Interface
- `[kaggle_runner.py:write_meta]`: Serializes run metadata dictionary `meta` to `/kaggle/working/run_meta.json`.
- `[kaggle_runner.py:sh]`: Executes shell command list via `subprocess.run` with `check=True`.
- `[kaggle_runner.py:version]`: Looks up installed package version strings using `importlib.metadata.version`.
- `[kaggle_runner.py:find_caltech_data_root]`: Searches directory trees under `/kaggle/input` for Caltech-256 dataset root.
- `[kaggle_runner.py:find_tier1_cache_sources]`: Discovers directory sources under `/kaggle/input` containing HDF5 cache archives (`*.h5`).

## Invariants
- `[kaggle_runner.py:RUN_CONFIG]`: Expects global dictionary `RUN_CONFIG` prepended by CI workflow containing keys `"github_run_id"`, `"repo"`, `"sha"`, `"ref"`, and `"command"`.
- `[kaggle_runner.py]`: When `cfg["command"]` references `TIER1_CACHE_DIR`, at least one directory containing `*.h5` must exist under `/kaggle/input`; raises `RuntimeError` if missing.

## Side Effects
- `[kaggle_runner.py:write_meta]`: Creates and overwrites `/kaggle/working/run_meta.json` with execution timing and status.
- `[kaggle_runner.py]`: Clones Git repository into `/tmp/smart-grouping` and checks out the target commit SHA.
- `[kaggle_runner.py]`: Installs Python dependencies using `pip install`.
- `[kaggle_runner.py]`: Executes arbitrary workload shell commands inside `/kaggle/working`.

## Verification
- Run: `pytest`
