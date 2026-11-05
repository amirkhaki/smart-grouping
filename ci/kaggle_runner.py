"""Kaggle Script Kernel Bootstrap Harness for smart-grouping.

This file forms the body of the Kaggle script kernel. The GitHub workflow
prepends a `RUN_CONFIG = {...}` dictionary containing repository, commit SHA,
and the target command to run, and pushes the kernel to Kaggle.

Outputs:
Artifacts left in /kaggle/working are downloaded by the workflow and uploaded
as GitHub Action artifacts.
"""

import datetime
import importlib.metadata as md
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

WORK = Path("/kaggle/working")
SRC = Path("/tmp/smart-grouping")
META = WORK / "run_meta.json"

cfg = RUN_CONFIG  # noqa: F821
meta = {
    "github_run_id": cfg["github_run_id"],
    "repo": cfg["repo"],
    "commit": cfg["sha"],
    "ref": cfg["ref"],
    "command": cfg["command"],
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "started",
}


def write_meta():
    META.write_text(json.dumps(meta, indent=2))


def sh(cmd, cwd=None):
    print("$", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def find_caltech_data_root(base=Path("/kaggle/input"), max_depth=5):
    """Find Caltech-256 directory under /kaggle/input."""
    found = []

    def visit(d, depth):
        if (d / "256_ObjectCategories").is_dir() or (d / "001.ak47").is_dir():
            target = d / "256_ObjectCategories" if (d / "256_ObjectCategories").is_dir() else d
            found.append(target)
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for c in children:
            visit(c, depth + 1)

    if base.exists():
        visit(base, 0)
    return found


def find_tier1_cache_sources(base=Path("/kaggle/input"), max_depth=6):
    """Find directories under /kaggle/input containing caltech256_elic_latents.h5."""
    found = []

    def visit(d, depth):
        if any(d.glob("*.h5")):
            found.append(d)
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(c for c in d.iterdir() if c.is_dir())
        except OSError:
            return
        for c in children:
            visit(c, depth + 1)

    if base.exists():
        visit(base, 0)
    return found


write_meta()
t0 = time.time()

try:
    # 1. Clone exact commit
    sh(["git", "clone", f"https://github.com/{cfg['repo']}.git", str(SRC)])
    sh(["git", "checkout", "--detach", cfg["sha"]], cwd=SRC)
    meta["commit_verified"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SRC, text=True
    ).strip()

    # 2. Discover attached inputs
    inputs = Path("/kaggle/input")
    meta["inputs"] = sorted(p.name for p in inputs.iterdir()) if inputs.exists() else []
    print("Attached inputs:", meta["inputs"], flush=True)

    # Caltech-256 data root
    caltech_roots = find_caltech_data_root()
    caltech_data_root = caltech_roots[0] if caltech_roots else None
    if caltech_data_root:
        print(f"Caltech data source: {caltech_data_root}", flush=True)
    meta["caltech_data_root"] = str(caltech_data_root) if caltech_data_root else None

    # Tier-1 HDF5 cache directory
    tier1_dirs = find_tier1_cache_sources()
    tier1_cache_dir = tier1_dirs[0] if tier1_dirs else None
    if tier1_cache_dir:
        print(f"Tier-1 cache source: {tier1_cache_dir}", flush=True)
        rm = tier1_cache_dir / "run_meta.json"
        if rm.is_file():
            try:
                j = json.loads(rm.read_text())
                meta["tier1_provenance"] = {
                    "commit": j.get("commit_verified") or j.get("commit"),
                    "command": j.get("command"),
                    "status": j.get("status"),
                }
            except Exception as e:
                meta["tier1_provenance_error"] = repr(e)
    meta["tier1_cache_dir"] = str(tier1_cache_dir) if tier1_cache_dir else None

    # Guard: command references TIER1_CACHE_DIR but none found
    if "TIER1_CACHE_DIR" in cfg["command"] and tier1_cache_dir is None:
        raise RuntimeError(
            "Command references $TIER1_CACHE_DIR but no HDF5 cache (*.h5) was found under /kaggle/input. "
            "Attach the smart-grouping-tier1-cache kernel as a kernel source."
        )

    # 3. Pip install dependencies
    try:
        sh([sys.executable, "-m", "pip", "install", "compressai==1.1.5", "gdown", "h5py", "timm", "ptflops", "thop", "iqadataset"])
    except Exception as e:
        print(f"Warning: compressai==1.1.5 install failed ({e}), trying generic compressai...", flush=True)
        sh([sys.executable, "-m", "pip", "install", "compressai", "gdown", "h5py", "timm", "ptflops", "thop", "iqadataset"])

    try:
        sh([sys.executable, "-m", "pip", "install", "--no-deps", "-e", str(SRC)])
    except Exception as e:
        print(f"Warning: pip install -e failed ({e}), relying on PYTHONPATH fallback", flush=True)

    # Record GPU and library versions
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    except FileNotFoundError:
        gpu = ""
    meta["gpu"] = gpu or None
    meta["torch"] = version("torch")
    meta["torchvision"] = version("torchvision")
    meta["h5py"] = version("h5py")

    # 4. Execute target command
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    if caltech_data_root and "CALTECH_DATA_ROOT" not in env:
        env["CALTECH_DATA_ROOT"] = str(caltech_data_root)
    if tier1_cache_dir and "TIER1_CACHE_DIR" not in env:
        env["TIER1_CACHE_DIR"] = str(tier1_cache_dir)

    print("$", cfg["command"], flush=True)
    subprocess.run(
        ["bash", "-eo", "pipefail", "-c", cfg["command"]],
        cwd=WORK,
        env=env,
        check=True,
    )
    meta["status"] = "ok"

except BaseException as e:
    meta["status"] = "failed"
    meta["error"] = repr(e)
    raise

finally:
    # Clean up checkpoint weights and temporary directories from working dir to prevent bloating artifacts
    shutil.rmtree(WORK / "checkpoints", ignore_errors=True)
    for p in WORK.glob("*.pth*"):
        try:
            p.unlink()
        except OSError:
            pass

    # Record output files and bytes
    out_files = 0
    out_bytes = 0
    for r, _, names in os.walk(WORK):
        for n in names:
            out_files += 1
            try:
                out_bytes += os.path.getsize(os.path.join(r, n))
            except OSError:
                pass
    meta["output_files"] = out_files
    meta["output_bytes"] = out_bytes
    print(f"Kernel output: {out_files} files, {out_bytes / 1e9:.2f} GB", flush=True)

    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["wall_seconds"] = round(time.time() - t0, 1)
    write_meta()
