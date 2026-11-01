"""Configuration constants, dataclasses, and registries for ELIC slice benchmarking."""

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple


@dataclass(frozen=True)
class SliceSpec:
    """Specification of a channel slice view."""
    name: str
    start_ch: int
    end_ch: int
    num_channels: int
    category: Literal["isolated", "cumulative"]
    description: str


# The 9 unique slice views (S0 serves as both isolated [0:16] and cumulative [0:16])
SLICE_SPECS: Dict[str, SliceSpec] = {
    "S0": SliceSpec("S0", 0, 16, 16, "isolated", "Hyperprior-conditioned base slice [0:16]"),
    "S1": SliceSpec("S1", 16, 32, 16, "isolated", "First contextual channel residual [16:32]"),
    "S2": SliceSpec("S2", 32, 64, 32, "isolated", "Mid-frequency channel residual [32:64]"),
    "S3": SliceSpec("S3", 64, 128, 64, "isolated", "Intermediate spatial-channel residual [64:128]"),
    "S4": SliceSpec("S4", 128, 320, 192, "isolated", "High-capacity high-frequency textural residual [128:320]"),
    "C1": SliceSpec("C1", 0, 32, 32, "cumulative", "Cumulative union of S0 + S1 [0:32]"),
    "C2": SliceSpec("C2", 0, 64, 64, "cumulative", "Cumulative union of S0 + S1 + S2 [0:64]"),
    "C3": SliceSpec("C3", 0, 128, 128, "cumulative", "Cumulative union of S0 + S1 + S2 + S3 [0:128]"),
    "C4": SliceSpec("C4", 0, 320, 320, "cumulative", "Full 320-channel latent representation [0:320]"),
}

SLICE_VIEW_NAMES: List[str] = list(SLICE_SPECS.keys())
ISOLATED_SLICES: List[str] = ["S0", "S1", "S2", "S3", "S4"]
CUMULATIVE_SLICES: List[str] = ["S0", "C1", "C2", "C3", "C4"]

# Quantization and Probe modes
QUANTIZATION_MODES: List[str] = ["quantized", "continuous"]
PROBE_ARCHITECTURES: List[str] = ["linear_probe", "latent_head"]

# ELIC Checkpoint Registry on Google Drive
@dataclass(frozen=True)
class CheckpointMeta:
    lmbda: float
    drive_id: str
    default_filename: str
    description: str


CHECKPOINT_REGISTRY: Dict[str, CheckpointMeta] = {
    "0.032": CheckpointMeta(
        lmbda=0.032,
        drive_id="1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc",
        default_filename="checkpoints/elic_0032.pth.tar",
        description="Primary balanced baseline for 36-run matrix",
    ),
    "0.004": CheckpointMeta(
        lmbda=0.004,
        drive_id="1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q",
        default_filename="checkpoints/elic_0004.pth.tar",
        description="Low-rate regime exhibiting dead-zone slice collapse",
    ),
    "0.008": CheckpointMeta(
        lmbda=0.008,
        drive_id="1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL",
        default_filename="checkpoints/elic_0008.pth.tar",
        description="Intermediate low-rate ablation checkpoint",
    ),
    "0.016": CheckpointMeta(
        lmbda=0.016,
        drive_id="1MWlYAmpHbWlGtG7MBBTPEew800grY5yC",
        default_filename="checkpoints/elic_0016.pth.tar",
        description="Intermediate mid-rate ablation checkpoint",
    ),
    "0.15": CheckpointMeta(
        lmbda=0.15,
        drive_id="1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL",
        default_filename="checkpoints/elic_0150.pth.tar",
        description="High-rate regime checkpoint",
    ),
    "0.45": CheckpointMeta(
        lmbda=0.45,
        drive_id="1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN",
        default_filename="checkpoints/elic_0450.pth.tar",
        description="Maximum fidelity regime checkpoint",
    ),
}

# Dataset & Partitioning Constants
NUM_CLASSES: int = 257  # 256 object categories + 257.clutter
IMAGE_SIZE: Tuple[int, int] = (256, 256)
LATENT_SPATIAL_DIM: Tuple[int, int] = (16, 16)
LATENT_TOTAL_CHANNELS: int = 320
TRAIN_SPLIT_RATIO: float = 0.80
AUGMENTATION_TARGET_K: int = 150  # Balanced samples per class in train split
EXPECTED_TOTAL_TRAIN_SAMPLES: int = 38550  # 257 classes * 150 samples
EXPECTED_BASE_TEST_SAMPLES: int = 6122
EXPECTED_TOTAL_ARCHIVE_SAMPLES: int = 44672

# Storage Constants
HDF5_CHUNK_SHAPE: Tuple[int, int, int, int] = (32, 320, 16, 16)
DEFAULT_CACHE_FILENAME: str = "caltech256_elic_latents.h5"

# Training Hyperparameters
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_EPOCHS: int = 50
DEFAULT_LR: float = 1e-3
DEFAULT_WEIGHT_DECAY: float = 1e-4
DEFAULT_DROPOUT: float = 0.3
LATENT_HEAD_STEM_CHANNELS: int = 128
DEFAULT_STEM_CHANNELS: List[int] = [128]
STEM_ABLATION_CHANNELS: List[int] = [8, 16, 32, 128]
