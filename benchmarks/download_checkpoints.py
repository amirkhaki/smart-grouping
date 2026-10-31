"""Automated downloader for pretrained ELIC checkpoints from Google Drive."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

from benchmarks.config import CHECKPOINT_REGISTRY, CheckpointMeta


def download_checkpoint(
    meta: CheckpointMeta,
    output_dir: Path,
    max_retries: int = 5,
    backoff_factor: float = 2.0,
) -> Path:
    """Download a single checkpoint via gdown with exponential backoff."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / Path(meta.default_filename).name

    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        print(f"[SKIP] Checkpoint already exists: {target_path} ({target_path.stat().st_size / (1024*1024):.1f} MB)")
        return target_path

    try:
        import gdown
    except ImportError:
        raise ImportError(
            "gdown is required for downloading checkpoints. Install with: pip install gdown"
        )

    url = f"https://drive.google.com/uc?id={meta.drive_id}"
    print(f"[DOWNLOAD] Fetching lambda={meta.lmbda} ({meta.drive_id}) -> {target_path}")

    wait_time = 2.0
    for attempt in range(1, max_retries + 1):
        try:
            downloaded = gdown.download(
                id=meta.drive_id,
                output=str(target_path),
                quiet=False,
                fuzzy=True,
            )
            if downloaded and Path(downloaded).exists() and Path(downloaded).stat().st_size > 1024 * 1024:
                print(f"[SUCCESS] Downloaded: {target_path} ({Path(downloaded).stat().st_size / (1024*1024):.1f} MB)")
                return target_path
            else:
                raise RuntimeError(f"Downloaded file {target_path} is missing or incomplete.")
        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download checkpoint lambda={meta.lmbda} after {max_retries} attempts: {e}"
                )
            time.sleep(wait_time)
            wait_time *= backoff_factor

    return target_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Download pretrained ELIC checkpoints from Google Drive.")
    parser.add_argument(
        "--lmbda",
        type=str,
        default="0.032",
        choices=list(CHECKPOINT_REGISTRY.keys()),
        help="Rate identifier lambda (default: 0.032)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all registered checkpoints in CHECKPOINT_REGISTRY",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Target directory for checkpoints (default: checkpoints)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)

    if args.all:
        for key, meta in CHECKPOINT_REGISTRY.items():
            print(f"\nProcessing lambda={key}...")
            download_checkpoint(meta, out_dir)
    else:
        meta = CHECKPOINT_REGISTRY[args.lmbda]
        download_checkpoint(meta, out_dir)


if __name__ == "__main__":
    main(sys.argv[1:])
