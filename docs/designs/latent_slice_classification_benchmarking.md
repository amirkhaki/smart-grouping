# Latent Slice Classification Benchmarking on Frozen ELIC

*   **Author:** @alice
*   **Status:** Proposed
*   **Date:** 2026-10-31

--------------------------------------------------------------------------------

## Overview

This project establishes an automated, reproducible benchmarking framework to evaluate the semantic classification utility of unevenly grouped latent slices from a pretrained, frozen ELIC image compression network on the Caltech-256 dataset. 

ELIC partitions its 320-channel latent representation into five uneven slices: 16, 16, 32, 64, and 192 channels. This framework probes each isolated slice and cumulative progressive groupings (16, 32, 64, 128, and 320 channels) across both minimal Linear Probes and capacity-normalized PreAct ResNet heads (`LatentHead`). Deduplicating the identical first slice (`0:16` isolated vs. `0:16` cumulative) yields **9 unique slice views**.

To decouple expensive neural feature extraction from rapid downstream classifier iteration, the system uses a 2-tier execution pipeline on Kaggle automated via GitHub Actions:
1. **Tier-1 (Cache Kernel)**: Prepares stratified, leakage-free augmented splits and extracts deterministic quantized ($\hat{y}$) and continuous ($y$) latents alongside per-slice bitrates into an indexed HDF5 archive (`float16`) on 2x Tesla T4 GPUs within Kaggle's 20 GB disk limit.
2. **Tier-2 (Evaluation Kernel)**: Attaches the Tier-1 cache as a kernel source, loads the archive, and trains all 36 classification heads (9 slice views $\times$ 2 probe architectures $\times$ 2 quantization states) **in parallel within a single shared training loop** on 2x Tesla T4 GPUs, publishing Rate-Accuracy Pareto curves and metric tables back to GitHub Actions as workflow artifacts.

## The Problem

Learned image compression architectures optimize an end-to-end rate-distortion objective ($\mathcal{L} = \lambda D + R$) focused strictly on pixel fidelity (MSE / MS-SSIM) and entropy coding rate (bpp). While uneven space-channel contextual grouping significantly improves rate-distortion performance, several critical representation questions remain unanswered:

1. **Uneven Channel Semantic Specialization**: Do early, narrow slices (e.g., 16 channels) capture compact global semantic structures while later, wide slices (192 channels) allocate capacity exclusively to high-frequency textural residuals, or is semantic information distributed throughout?
2. **Progressive Semantic Accumulation**: In progressive transmission, how does classification accuracy scale as successive bitstream slices are received and decoded?
3. **The Quantization Tax & Dead-Zone Collapse**: Quantization maps continuous features onto discrete grids for entropy coding. At low rates, high-index slices often experience severe dead-zone collapse (where $\text{round}(y - \mu) \equiv 0$). What is the exact quantitative degradation in linear separability and non-linear classification accuracy caused by this quantization step?
4. **Context-Conditioned Quantization Semantics**: In ELIC's Space-Channel Context Model, slice $k$ is quantized via mean-shifted rounding $\hat{y}^{(k)} = \text{round}(y^{(k)} - \mu^{(k)}) + \mu^{(k)}$, where $\mu^{(k)}$ is predicted from the hyperprior $\hat{z}$ and preceding decoded slices $\hat{y}^{(<k)}$. Evaluating isolated $\hat{y}^{(k)}$ measures the reconstructed latent channel allocation entering $g_s$ rather than an independently decodable transmission packet.
5. **Platform & Infrastructure Bottlenecks**: Extracting latents through the ELIC analysis transform and sequential context model is computationally intensive. Coupling feature extraction with classifier training wastes GPU resources and exceeds ephemeral execution and storage limits. The benchmarking pipeline must execute within Kaggle's 20 GB disk quota and 12-hour session constraints while maintaining full CI automation via GitHub Actions.

## Goals

*   **Individual Slice Specialization**: Measure Top-1, Top-5, and Macro-F1 accuracy across all 5 isolated slices (`[0:16]`, `[16:32]`, `[32:64]`, `[64:128]`, and `[128:320]`) of the fully decoded $\hat{y}$ entering $g_s$.
*   **Cumulative Progressive Scaling & Pareto Analysis**: Measure Top-1 and Top-5 accuracy across all cumulative slice unions (`0:16`, `0:32`, `0:64`, `0:128`, and `0:320`), plotting Rate-Accuracy Pareto curves against cumulative decodable bitrate:
    $$\text{bpp}_k = R(\hat{z}) + \sum_{i=1}^k R(\hat{y}^{(i)})$$
    Tier-1 caches per-image float bitrates ($R(\hat{z})$ and each $R(\hat{y}^{(i)})$) to compute exact empirical test-set bitrates.
*   **Dual Probe Hierarchy**:
    *   **Minimal Linear Probe**: `GAP -> BatchNorm1d(C_in) -> Linear(C_in, 257)` to assess linear feature separability.
    *   **Capacity-Normalized LatentHead**: `1x1 Conv stem (C_in -> 128) -> PreAct ResNet blocks -> BatchNorm2d -> ReLU -> GAP -> Dropout(0.3) -> Linear(128, 257)`. The post-block `BatchNorm2d -> ReLU` guarantees non-linearity across the residual highway, while the $1 \times 1$ stem equalizes residual trunk capacity across slices.
*   **Parallel Multi-Head Tier-2 Optimization**: Train all 36 probe configurations concurrently in a single DataLoader pass over cached HDF5 features, reducing disk I/O by $36\times$ and completing evaluation in under 15–20 minutes.
*   **Quantization Penalty Ablation Across Rate Regimes**: Benchmark both unquantized continuous latents ($y$) and quantized latents ($\hat{y}$) across two distinct rate checkpoints:
    *   $\lambda = 0.032$: Primary balanced rate-distortion regime.
    *   $\lambda = 0.004$: Low-rate regime exhibiting dead-zone slice collapse.
*   **Strict Leakage-Free Data Protocol**: 
    *   Images standardized to $256 \times 256$ center-crops (strictly divisible by 64), producing unpadded $16 \times 16 \times 320$ latents.
    *   Sample-level stratified splitting (80% train / 20% test across all 257 Caltech-256 classes) *prior* to augmentation.
    *   Offline training-split augmentation balancing classes up to $K_{\text{target}} = 150$ samples per class.
*   **FP16 Cache Storage Architecture**: Cache latents in `float16` ($320 \times 16 \times 16 \times 2\text{ bytes} = 160\text{ KB}$ per tensor), keeping the complete augmented cache (44,672 samples $\times$ 320 KB + metadata) at **14.63 GB**, safely under Kaggle's 20 GB disk limit.
*   **Two-Tier Kaggle CI Automation**: Automated execution via GitHub Actions workflows (`kaggle-tier1.yml` and `kaggle-tier2.yml`) and the Kaggle API following the `FoundationIQA` composite action architecture.

## Non-Goals

*   **Backbone Fine-Tuning or Joint Training**: The ELIC compression model weights remain strictly frozen. No classification loss gradients backpropagate into $g_a$ or the context model.
*   **Modifying Core Compression Code**: Existing files in `/Projects/smart-grouping` (`Network.py`, `train.py`, `Inference.py`, `updata.py`) remain untouched. All benchmark code, extractors, and CI runners reside in self-contained directories (`benchmarks/`, `ci/`, `.github/`).
*   **Real-Time Bitstream Decoding During Training**: Arithmetic decoding on CPU from raw binary strings is bypassed during head training; Tier-1 applies deterministic evaluation quantization (`torch.round(y - means) + means`), saving decoded coefficients directly.
*   **Generalizing to Other Vision Tasks**: Object detection, semantic segmentation, and generative restoration are out of scope.

## Open Questions (Resolved)

*   **Resolved (Augmentation Target Depth & Storage Budget)**: 
    *   *Decision*: Use $K_{\text{target}} = 150$ samples per class in the training split with `float16` storage format.
    *   *Arithmetic*: 38,550 train samples (257 classes $\times$ 150) + 6,122 test samples = **44,672 total samples**. At $160\text{ KB}$ per tensor ($y$ and $\hat{y}$ combined = $320\text{ KB}$), total archive size is **14.63 GB**, fitting comfortably within the 15 GB cache budget and Kaggle's 20 GB limit.
*   **Resolved (Checkpoint Rate Coverage)**:
    *   *Decision*: Primary sweep uses $\lambda = 0.032$; a comparative sweep uses $\lambda = 0.004$ to evaluate the impact of dead-zone slice collapse at low bitrates via a workflow parameter `lambda_ckpt: "0.032,0.004"`.

--------------------------------------------------------------------------------

## Critical User Journeys (CUJs)

### CUJ 1: Tier-1 Latent Extraction & Caching on Kaggle

*   **Given**: The Kaggle dataset `jessicali9530/caltech256` is mounted at `/kaggle/input/caltech256/256_ObjectCategories`, repository code is checked out at target commit SHA, and 2x Tesla T4 GPUs are active.
*   **When**: Developer or CI triggers the `kaggle-tier1.yml` workflow via GitHub Actions `workflow_dispatch`.
*   **Then (Happy Path)**:
    1. Downloads pretrained ELIC checkpoint(s) (`lambda=0.032` and/or `0.004`) automatically via `gdown` using Google Drive IDs.
    2. Partitions images deterministically into stratified 80% train (24,485 base images) and 20% test (6,122 base images) splits.
    3. Augments only the training split to equalize all 257 classes to $K_{\text{target}}=150$ samples (total 38,550 train images).
    4. Distributes the frozen ELIC model across both T4 GPUs via `CustomDataParallel` (batch size 64, 32 per GPU) to maximize extraction throughput.
    5. Extracts deterministic continuous latents $y$, quantized latents $\hat{y}$, hyperprior rate $R(\hat{z})$, and per-slice bitrates $R(\hat{y}_k)$ via `ELICFeatureExtractor`.
    6. Serializes all features into an indexed HDF5 archive (`caltech256_elic_latents.h5`, 14.63 GB in `float16`) and `run_meta.json` inside `/kaggle/working`.
    7. GitHub Actions runner filters out `.h5` files and downloads only `run_meta.json` and diagnostic logs as artifacts, leaving the HDF5 archive on Kaggle under slug `smart-grouping-tier1-cache`.
*   **Alternate / Failure Paths**:
    *   *gdown Rate Limit*: If Google Drive returns HTTP 429/503, the downloader retries up to 3 times with exponential backoff (10s, 30s, 60s) before terminating with an actionable error.
    *   *Disk Exhaustion Protection*: If free disk in `/kaggle/working` drops below 2.0 GB during caching, extraction aborts with a clean error message rather than corrupting the HDF5 structure.

### CUJ 2: Tier-2 Multi-Head Parallel Training & Pareto Curve Generation

*   **Given**: Tier-1 kernel has finished running and its output is available on Kaggle as `smart-grouping-tier1-cache`.
*   **When**: Developer or CI triggers the `kaggle-tier2.yml` workflow via GitHub Actions `workflow_dispatch`.
*   **Then (Happy Path)**:
    1. Mounts `smart-grouping-tier1-cache` via `kernel_sources` under `/kaggle/input/smart-grouping-tier1-cache`.
    2. Verifies HDF5 file integrity (checks key presence and element count = 44,672).
    3. Instantiates all 36 classification heads within `MultiHeadClassifierContainer` wrapped in `torch.nn.DataParallel`.
    4. Executes a single unified training loop for 50 epochs over cached features on-the-fly (SWMR mode) using AdamW (`lr=1e-3`) and Cosine Annealing, training all 36 heads concurrently on 2x T4 GPUs in ~10–12 minutes.
    5. Evaluates the unaugmented test split, calculating Top-1, Top-5, and Macro-F1 accuracy, as well as exact empirical test-set bitrates:
       $$\text{bpp}_k = R(\hat{z}) + \sum_{i=1}^k R(\hat{y}^{(i)})$$
    6. Generates `master_results.csv`, `metrics.json`, Rate-Accuracy Pareto plots, and quantization delta comparisons in `/kaggle/working`.
    7. Downloads all evaluation metrics and plots to GitHub Actions as workflow run artifacts.
*   **Alternate / Failure Paths**:
    *   *Corrupted Cache Mount*: If `caltech256_elic_latents.h5` fails checksum or has missing keys, Tier-2 aborts immediately with exit code 2 and outputs a descriptive failure JSON.

### CUJ 3: Local Prototyping & Verification Run

*   **Given**: A small local subset of Caltech-256 (or mock HDF5 cache) is present on developer workstation.
*   **When**: Developer runs `python -m benchmarks.train_probe --cache-path <path> --epochs 1 --dry-run`.
*   **Then**: The multi-head trainer executes 1 training epoch and 1 evaluation step across all 36 heads, verifying input slicing boundaries, tensor dimensions, loss aggregation, and metric logging without error.

--------------------------------------------------------------------------------

## Technical Plan (Architecture)

### 1. Overall System Architecture

The benchmarking architecture is completely decoupled into two pipeline stages connected through an immutable HDF5 feature cache, orchestrated by a GitHub Actions CI driver:

```text
+--------------------------------------------------------------------------------------------------+
|                                  GITHUB ACTIONS CI ORCHESTRATION                                 |
|                                                                                                  |
|   .github/workflows/kaggle-tier1.yml                .github/workflows/kaggle-tier2.yml            |
|            |                                                 |                                   |
|            v (Pushes kernel bundle & polls API)              v (Attaches Tier-1 output & polls)  |
+------------|-------------------------------------------------|-----------------------------------+
             |                                                 |
             v                                                 v
+------------------------------------------+     +-------------------------------------------------+
|     KAGGLE TIER-1 (CACHE KERNEL)         |     |        KAGGLE TIER-2 (BENCHMARK KERNEL)         |
|  Slug: smart-grouping-tier1-cache        |     |  Slug: smart-grouping-tier2-eval                |
|  Hardware: 2x Tesla T4 GPUs              |     |  Hardware: 2x Tesla T4 GPUs                     |
|                                          |     |                                                 |
|  Input:                                  |     |  Input:                                         |
|    - /kaggle/input/caltech256/           |     |    - /kaggle/input/smart-grouping-tier1-cache/  |
|    - gdown Checkpoint (0.032 / 0.004)    |     |      (caltech256_elic_latents.h5)               |
|                                          |     |                                                 |
|  Execution:                              |     |  Execution:                                     |
|    1. Stratified Split (80/20)           |     |    1. Open HDF5 (on-the-fly, SWMR=True)         |
|    2. Offline Augmentation (Train only)  |     |    2. Initialize 36 Parallel Heads               |
|    3. CustomDataParallel Frozen Forward: |     |    3. DataParallel 50-Epoch Training Loop       |
|       - Continuous y in R^(320x16x16)    |     |    4. Test Set Evaluation & BPP Aggregation     |
|       - Quantized y_hat in R^(320x16x16) |     |                                                 |
|       - Empirical BPPs: R(z), R(y_k)     |     |  Output:                                        |
|                                          |     |    - /kaggle/working/master_results.csv         |
|  Output:                                 |     |    - /kaggle/working/summary_metrics.json       |
|    - /kaggle/working/                    |     |    - /kaggle/working/pareto_curves.png          |
|      caltech256_elic_latents.h5 (14.63GB)|     |                                                 |
+------------------------------------------+     +-------------------------------------------------+
```

### 2. Latent Extraction Engine for Untouched Network.py

In `Network.py:TestModel`, continuous latents $y$ and quantized latents $\hat{y}$ are internal local variables within `forward()` and `inference()`. To strictly adhere to the Non-Goal of leaving `Network.py` untouched, a dedicated wrapper `benchmarks/elic_extractor.py:ELICFeatureExtractor` is implemented:

```python
class ELICFeatureExtractor(nn.Module):
    """Encapsulates frozen TestModel to extract internal representations without code modification."""
    def __init__(self, model: TestModel):
        super().__init__()
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        # 1. Continuous latent from analysis transform
        y = self.model.g_a(x)
        B, C, H, W = y.size()
        num_pixels = x.size(0) * x.size(2) * x.size(3)

        # 2. Hyperprior forward pass
        z = self.model.h_a(y)
        z_hat, z_likelihoods = self.model.entropy_bottleneck(z)
        z_offset = self.model.entropy_bottleneck._get_medians()
        z_hat = torch.round(z - z_offset) + z_offset
        bpp_z = torch.log(z_likelihoods).sum() / (-math.log(2) * num_pixels)

        # 3. Hyperprior synthesis (spatial parameter prior)
        latent_means, latent_scales = self.model.h_s(z_hat).chunk(2, 1)

        # 4. Sequential Context Model & Slice Quantization Loop
        anchor = torch.zeros_like(y)
        non_anchor = torch.zeros_like(y)
        anchor[:, :, 0::2, 0::2] = y[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y[:, :, 1::2, 0::2]

        y_slices = torch.split(y, self.model.groups[1:], 1)
        anchor_split = torch.split(anchor, self.model.groups[1:], 1)
        non_anchor_split = torch.split(non_anchor, self.model.groups[1:], 1)
        ctx_params_anchor_split = torch.split(torch.zeros(B, C * 2, H, W, device=x.device),
                                              [2 * i for i in self.model.groups[1:]], 1)

        y_hat_slices = []
        bpp_y_slices = []

        for slice_index, y_slice in enumerate(y_slices):
            if slice_index == 0:
                support = torch.concat([latent_means, latent_scales], dim=1)
            else:
                support_slices = (y_hat_slices[0] if slice_index == 1 
                                  else torch.concat([y_hat_slices[0], y_hat_slices[slice_index-1]], dim=1))
                support_slices_ch = self.model.cc_transforms[slice_index-1](support_slices)
                mean_ch, scale_ch = support_slices_ch.chunk(2, 1)
                support = torch.concat([mean_ch, scale_ch, latent_means, latent_scales], dim=1)

            # Anchor step
            means_a, scales_a = self.model.ParamAggregation[slice_index](
                torch.concat([ctx_params_anchor_split[slice_index], support], dim=1)).chunk(2, 1)
            y_a_quant = torch.round(anchor_split[slice_index] - means_a) + means_a
            y_a_quant[:, :, 0::2, 1::2] = 0
            y_a_quant[:, :, 1::2, 0::2] = 0

            # Non-anchor step
            masked_ctx = self.model.context_prediction[slice_index](y_a_quant)
            means_na, scales_na = self.model.ParamAggregation[slice_index](
                torch.concat([masked_ctx, support], dim=1)).chunk(2, 1)
            y_na_quant = torch.round(non_anchor_split[slice_index] - means_na) + means_na
            y_na_quant[:, :, 0::2, 0::2] = 0
            y_na_quant[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_a_quant + y_na_quant
            y_hat_slices.append(y_hat_slice)

            # Likelihood & Bitrate computation
            means_full = torch.zeros_like(y_slice)
            scales_full = torch.zeros_like(y_slice)
            means_full[:, :, 0::2, 0::2] = means_a[:, :, 0::2, 0::2]
            means_full[:, :, 1::2, 1::2] = means_a[:, :, 1::2, 1::2]
            means_full[:, :, 0::2, 1::2] = means_na[:, :, 0::2, 1::2]
            means_full[:, :, 1::2, 0::2] = means_na[:, :, 1::2, 0::2]
            scales_full[:, :, 0::2, 0::2] = scales_a[:, :, 0::2, 0::2]
            scales_full[:, :, 1::2, 1::2] = scales_a[:, :, 1::2, 1::2]
            scales_full[:, :, 0::2, 1::2] = scales_na[:, :, 0::2, 1::2]
            scales_full[:, :, 1::2, 0::2] = scales_na[:, :, 1::2, 0::2]

            _, slice_likelihood = self.model.gaussian_conditional(y_slice, scales_full, means=means_full)
            bpp_slice = torch.log(slice_likelihood).sum() / (-math.log(2) * num_pixels)
            bpp_y_slices.append(bpp_slice.item())

        y_hat = torch.cat(y_hat_slices, dim=1)
        return {
            "y": y,
            "y_hat": y_hat,
            "bpp_z": bpp_z.item(),
            "bpp_slices": bpp_y_slices,
        }
```

### 3. Data Partitioning & Storage Schema (Tier-1)

#### Stratified Split & Augmentation Pipeline
1. **Raw Dataset Layout**: Caltech-256 contains 256 object folders plus `257.clutter` (total 30,607 images).
2. **Deterministic Hash Splitting**:
   - Each image filename and class ID are hashed to assign samples into 80% train (24,485 base images) and 20% test (6,122 base images).
   - Test partition remains strictly unaugmented, resized shortest-side to 256 and center-cropped to $256 \times 256$.
3. **Class-Balanced Offline Augmentation**:
   - For each class $c \in \{1, \dots, 257\}$, if training base count $N_{\text{train}, c} < K_{\text{target}}$ ($K_{\text{target}}=150$), synthetic views are created using random horizontal flip ($p=0.5$), rotation ($\pm 10^\circ$), and random crop ratio ($[0.85, 1.0]$) resized to $256 \times 256$. Classes with $\ge 150$ samples retain their natural distribution.
   - Total train samples: $257 \times 150 = 38,550$.
   - Total archive samples: $38,550 + 6,122 = \mathbf{44,672\text{ images}}$.

#### HDF5 Storage Archive Schema (`caltech256_elic_latents.h5`)
*   `/latents_quantized`: Array of shape `(44672, 320, 16, 16)`, `dtype=float16`, chunk size `(32, 320, 16, 16)`.
*   `/latents_continuous`: Array of shape `(44672, 320, 16, 16)`, `dtype=float16`, chunk size `(32, 320, 16, 16)`.
*   `/labels`: Array of shape `(44672,)`, `dtype=int64` (class integer `0..256`).
*   `/is_train`: Array of shape `(44672,)`, `dtype=bool` (`True` for train, `False` for test).
*   `/bitrates_bpp`: Array of shape `(44672, 6)`, `dtype=float32` ($[R(\hat{z}), R(\hat{y}_0), R(\hat{y}_1), R(\hat{y}_2), R(\hat{y}_3), R(\hat{y}_4)]$).
*   `attrs`: Checkpoint filename, $\lambda$ value, class index mapping table, creation timestamp.

*Exact Storage Arithmetic*:
$$44,672 \times (320 \times 16 \times 16 \times 2 \times 2\text{ bytes} + 24\text{ bytes}) \approx \mathbf{14.63\text{ GB}}$$
Fits inside Kaggle's 20 GB `/kaggle/working` quota with 5.37 GB margin. Chunk size `(32, 320, 16, 16)` guarantees optimal sequential block reading.

### 4. Latent Slicing & Probing Matrix (Tier-2)

#### The 9 Unique Slice Views
| View Identifier | Category | Channel Range | Channels ($C_{\text{in}}$) | Representation Role |
| :--- | :--- | :--- | :--- | :--- |
| **S0** | Isolated | `[0:16]` | 16 | Hyperprior-conditioned base slice |
| **S1** | Isolated | `[16:32]` | 16 | First contextual channel residual |
| **S2** | Isolated | `[32:64]` | 32 | Mid-frequency channel residual |
| **S3** | Isolated | `[64:128]` | 64 | Intermediate spatial-channel residual |
| **S4** | Isolated | `[128:320]` | 192 | High-capacity high-frequency textural residual |
| **C1** | Cumulative | `[0:32]` | 32 | Union of S0 + S1 |
| **C2** | Cumulative | `[0:64]` | 64 | Union of S0 + S1 + S2 |
| **C3** | Cumulative | `[0:128]` | 128 | Union of S0 + S1 + S2 + S3 |
| **C4** | Cumulative | `[0:320]` | 320 | Full 320-channel latent representation |

*(Note: Isolated `[0:16]` and Cumulative `[0:16]` are identical and deduplicated as S0).*

#### Probe Head Specifications
1. **Minimal Linear Probe (`LinearProbe`)**:
   ```text
   Input: (B, C_in, 16, 16)
     │
     ▼
   nn.AdaptiveAvgPool2d((1, 1))  ──►  (B, C_in, 1, 1)
     │
     ▼
   nn.Flatten()                  ──►  (B, C_in)
     │
     ▼
   nn.BatchNorm1d(C_in)          ──►  (B, C_in)
     │
     ▼
   nn.Linear(C_in, 257)          ──►  (B, 257) Logits
   ```

2. **Capacity-Normalized PreAct ResNet Head (`LatentHead`)**:
   ```text
   Input: (B, C_in, 16, 16)
     │
     ▼
   Stem: Conv2d(C_in, 128, kernel=1, bias=False) -> BatchNorm2d(128) -> ReLU()
     │
     ▼
   Body:
     ├─► PreActBlock(128, 128, stride=1)  (16x16)
     ├─► PreActBlock(128, 128, stride=1)  (16x16)
     └─► PreActBlock(128, 128, stride=2)  (8x8, strided downsample)
     │
     ▼
   Neck: BatchNorm2d(128) -> ReLU()
     │
     ▼
   Head:
     ├─► nn.AdaptiveAvgPool2d((1, 1))
     ├─► nn.Flatten()
     ├─► nn.Dropout(p=0.3)
     └─► nn.Linear(128, 257)              ──►  (B, 257) Logits
   ```

   *PreActBlock Implementation Contract*:
   ```python
   class PreActBlock(nn.Module):
       def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
           super().__init__()
           self.b1 = nn.BatchNorm2d(in_ch)
           self.c1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
           self.b2 = nn.BatchNorm2d(out_ch)
           self.c2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
           self.shortcut = (
               nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
               if (stride != 1 or in_ch != out_ch)
               else nn.Identity()
           )

       def forward(self, x: torch.Tensor) -> torch.Tensor:
           o = F.relu(self.b1(x))
           sc = self.shortcut(o if isinstance(self.shortcut, nn.Conv2d) else x)
           return self.c2(F.relu(self.b2(self.c1(o)))) + sc
   ```

#### Multi-Head Joint Training Engine
Rather than launching 36 separate training scripts, Tier-2 instantiates a single `MultiHeadClassifierContainer` holding all 36 heads:
$$\mathcal{L}_{\text{total}} = \sum_{v \in \text{Views}} \sum_{a \in \{\text{Linear}, \text{LatentHead}\}} \sum_{q \in \{y, \hat{y}\}} \text{CrossEntropyLoss}\Big(h_{v,a,q}\big(\text{slice}(\mathbf{y}_q, v)\big), y_{\text{label}}\Big)$$
*   **Memory Safeguard**: Per-GPU batch size is set to 64 (effective batch size 128 across 2x T4 GPUs), keeping peak activation memory under 9 GB VRAM per GPU.
*   **I/O Concurrency**: Dataset reads HDF5 lazily using `swmr=True` per worker process without loading the file into system RAM.
*   **Optimizer**: Single AdamW optimizer (`lr=1e-3`, `weight_decay=1e-4`) with Cosine Annealing learning rate schedule over 50 epochs.

### 5. Automated Pretrained Checkpoint Ingestion via gdown

Pretrained ELIC weights are hosted on Google Drive as documented in `README.md`. A dedicated downloader module (`benchmarks/download_checkpoints.py`) resolves target checkpoints automatically using `gdown`:

| Rate Identifier ($\lambda$) | Google Drive File ID | Target Checkpoint Path | Purpose in Benchmark |
| :--- | :--- | :--- | :--- |
| **0.032** | `1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc` | `checkpoints/elic_0032.pth.tar` | Primary balanced baseline for full 36-run matrix |
| **0.004** | `1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q` | `checkpoints/elic_0004.pth.tar` | Low-rate regime demonstrating dead-zone slice collapse |
| **0.008** | `1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL` | `checkpoints/elic_0008.pth.tar` | Intermediate rate ablation (optional) |
| **0.016**\* | `1MWlYAmpHbWlGtG7MBBTPEew800grY5yC` | `checkpoints/elic_0016.pth.tar` | Intermediate rate ablation (optional) |
| **0.15** | `1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL` | `checkpoints/elic_0150.pth.tar` | High-rate regime (optional) |
| **0.45** | `1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN` | `checkpoints/elic_0450.pth.tar` | Maximum fidelity regime (optional) |

*\*Note: In upstream `README.md` line 33, the anchor text displays `[0.0016]` due to a typographical error, but maps to $\lambda=0.016$.*

### 6. Dual-GPU (2x Tesla T4) Parallel Acceleration

Kaggle GPU kernels provide two NVIDIA Tesla T4 GPUs (16 GB VRAM each). The framework utilizes both GPUs across both execution tiers:

1. **Tier-1 Feature Extraction Throughput**:
   - The extraction model is wrapped in `CustomDataParallel` (reused from `train.py:CustomDataParallel` with custom `__getattr__` delegation to preserve access to submodules and attributes).
   - Batch size is scaled to 64 (32 samples per GPU).
   - Feature extraction across 44,672 augmented images completes in $\approx 45\text{ minutes}$.

2. **Tier-2 Multi-Head Training Throughput**:
   - The `MultiHeadClassifierContainer` is distributed across both GPUs using `torch.nn.DataParallel`.
   - Effective batch size is 128 (64 samples per GPU).
   - 50 epochs over 38,550 train samples execute in $\approx 10\text{--}12\text{ minutes}$ total.

### 7. CI/CD Orchestration via GitHub Actions & Kaggle API

Following `FoundationIQA`:
1. **Composite Action (`.github/actions/kaggle-run/action.yml`)**:
   - Ingests `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN`.
   - Packages `ci/kaggle_runner.py` with an embedded `RUN_CONFIG` dictionary specifying target repo commit SHA and CLI command.
   - Pushes `kernel-metadata.json` to Kaggle via Kaggle CLI.
   - Polls `kaggle kernels status <owner>/<slug>` in 30-second increments with timeout handling.
   - **Runner Disk Exhaustion Guard**: Tier-1 workflow explicitly passes `artifact_paths: "out/run_meta.json\nout/*.log"`. A post-download step in the action removes `*.h5` from the runner's local checkout prior to artifact publishing, ensuring the runner does not exceed its 14 GB disk limit.
2. **Workflows**:
   - `kaggle-tier1.yml`: Dispatches `smart-grouping-tier1-cache` with dataset source `jessicali9530/caltech256`.
   - `kaggle-tier2.yml`: Dispatches `smart-grouping-tier2-eval` with kernel source `owner/smart-grouping-tier1-cache`.

--------------------------------------------------------------------------------

## Alternatives Considered

### 1. Cache Storage Format: HDF5 vs. Zarr vs. WebDataset vs. Raw Shards
*   **Alternative A (Raw .npy Shards)**: Saving individual `.npy` files per image.
    *   *Rejected*: Storing 44,672 files violates Kaggle's strict output quota (capped at ~500 output files per kernel) and degrades filesystem inode performance.
*   **Alternative B (WebDataset / TAR Shards)**: Writing `.tar` archives of NumPy tensors.
    *   *Rejected*: WebDataset is optimized for sequential streaming, but makes random-access slicing across arbitrary channel slices (`[32:64]`, `[128:320]`) inefficient without extracting tensors to disk.
*   **Alternative C (Zarr)**: Directory-based chunked array storage.
    *   *Rejected*: Creates thousands of chunk files within directories, causing zip/download overhead in Kaggle.
*   **Chosen (Single HDF5 File)**: Exactly one file in `/kaggle/working`, native multi-dimensional chunking, SWMR support, and zero Kaggle file-count overhead.

### 2. Multi-Head Training: Joint Container vs. Sequential Probe Scripts
*   **Alternative A (Sequential Independent Scripts)**: Running 36 separate training scripts one after another.
    *   *Rejected*: Requires 36 passes through the 14.6 GB dataset ($36 \times 50 = 1,800\text{ epochs}$ of disk reads), taking over 12 hours and exceeding Kaggle's kernel timeout.
*   **Chosen (Joint 36-Head Container)**: Evaluates all 36 heads in a single DataLoader pass per epoch. Cuts I/O by $36\times$ and completes all 36 runs in ~12 minutes.

### 3. Augmentation Strategy: Offline Image-Space vs. Online Latent-Space
*   **Alternative A (Online Latent-Space Augmentation)**: Augmenting cached latents directly (e.g. random channel masking or spatial noise).
    *   *Rejected*: Random noise in latent space does not correspond to realistic geometric transformations in pixel space and risks violating the nonlinear manifold of $g_a$.
*   **Chosen (Offline Image-Space Augmentation in Tier-1)**: Applies standard affine and photometric transformations in pixel space before passing through frozen $g_a$, ensuring all cached latents represent authentic compressed images.

### 4. Infrastructure Topology: Two-Tier Decoupled Kernels vs. Monolithic Kernel
*   **Alternative A (Monolithic Single Kaggle Notebook)**: Running feature extraction and head training in a single script.
    *   *Rejected*: Re-running classifier experiments, altering head learning rates, or debugging evaluation metrics would require paying the 45-minute GPU feature extraction penalty on every invocation.
*   **Chosen (Two-Tier Decoupled Kernels)**: Extraction runs once in Tier-1; rapid iteration occurs in Tier-2 attached via `kernel_sources`.

--------------------------------------------------------------------------------

## Non-Functional Requirements & Risk Mitigations

### 1. Kaggle Working Disk Limits (20 GB Limit)
*   **Risk**: Archive size exceeding 20 GB triggers Kaggle kernel eviction.
*   **Mitigation**: Latents are strictly stored in `float16` format. Total archive size is mathematically bounded to **14.63 GB**. Intermediate image crops are processed in streaming batches and never written to disk.

### 2. GitHub Actions Ephemeral Runner Disk Exhaustion (14 GB Limit)
*   **Risk**: Default `kaggle kernels output` downloads the entire `/kaggle/working` directory, crashing GitHub Actions runners.
*   **Mitigation**: The runner harness configures `artifact_paths` to capture only `out/run_meta.json` and `out/*.log`, and explicitly deletes `.h5` files from the runner checkout prior to artifact creation.

### 3. Dead-Zone Slice Collapse at Low Bitrates ($\lambda=0.004$)
*   **Risk**: When $\lambda=0.004$, slices S3 and S4 collapse to zero across almost all spatial coordinates, producing zero variance and causing `BatchNorm` division by zero.
*   **Mitigation**: All `BatchNorm1d` and `BatchNorm2d` layers maintain `eps=1e-5`. Training logs monitor per-slice activation variance $\text{Var}(\hat{y}_k)$ to explicitly report dead-zone collapse.

### 4. Google Drive Checkpoint Download Throttling
*   **Risk**: `gdown` encounters HTTP 429 rate limits when fetching weights.
*   **Mitigation**: `benchmarks/download_checkpoints.py` implements MD5 checksum verification to skip existing files, and applies exponential backoff retries.

### 5. Multi-Process DataLoader HDF5 Concurrency
*   **Risk**: Multiple PyTorch DataLoader workers reading an open `h5py.File` trigger race conditions or segfaults.
*   **Mitigation**: `HDF5LatentDataset` implements lazy file opening inside `__getitem__` (or in `worker_init_fn`) using Single-Writer Multiple-Reader (`swmr=True`) mode.

--------------------------------------------------------------------------------

## Detailed Implementation Plan

### Target Scope Table

| Target File Path | Action | Scope / Module Role | Interfaces / Dependencies |
| :--- | :--- | :--- | :--- |
| `benchmarks/__init__.py` | Create | Package initialization | None |
| `benchmarks/config.py` | Create | Central configuration dataclasses | Dataclasses for paths, slice windows, probe params |
| `benchmarks/download_checkpoints.py` | Create | Automated `gdown` checkpoint downloader | Resolves Drive IDs, downloads to `checkpoints/` |
| `benchmarks/elic_extractor.py` | Create | Non-invasive feature extraction wrapper | Wraps `TestModel`, exports $(y, \hat{y}, R(z), R(y_k))$ |
| `benchmarks/datasets/caltech256.py` | Create | Caltech-256 mounting, splitting & augmentation | Stratified 80/20 split, $K=150$ train augmentation |
| `benchmarks/datasets/hdf5_dataset.py` | Create | Thread-safe HDF5 PyTorch Dataset | Reads `float16` HDF5 in SWMR mode, yields float32 |
| `benchmarks/cache_latents.py` | Create | Tier-1 latent extraction CLI entrypoint | Dual-GPU `CustomDataParallel` extraction to HDF5 |
| `benchmarks/models/linear_probe.py` | Create | Minimal linear probe head | `GAP -> BatchNorm1d -> Linear(C, 257)` |
| `benchmarks/models/latent_head.py` | Create | PreAct ResNet probe head | `1x1 Stem -> PreActBlocks -> Neck -> GAP -> Linear` |
| `benchmarks/models/multi_head_container.py` | Create | Joint 36-head parallel model container | Evaluates 36 heads, computes aggregated loss |
| `benchmarks/train_probe.py` | Create | Tier-2 parallel training CLI entrypoint | Dual-GPU multi-head training, validation & Pareto |
| `benchmarks/evaluate.py` | Create | Metrics evaluation & visualization | Generates `master_results.csv`, Pareto plots |
| `ci/kaggle_runner.py` | Create | Kaggle script kernel bootstrap harness | Clones commit SHA, runs commands, saves meta |
| `.github/actions/kaggle-run/action.yml` | Create | Composite GitHub Action for Kaggle API | Pushes kernel, polls status, downloads outputs |
| `.github/workflows/kaggle-tier1.yml` | Create | Tier-1 caching GitHub Actions workflow | Dispatches `smart-grouping-tier1-cache` |
| `.github/workflows/kaggle-tier2.yml` | Create | Tier-2 evaluation GitHub Actions workflow | Dispatches `smart-grouping-tier2-eval` |

### Phased Implementation Plan

#### Phase 1: Checkpoint Downloader & Non-Invasive Extractor
*   **Goal**: Implement `benchmarks/download_checkpoints.py` and `benchmarks/elic_extractor.py`.
*   **Checkpoint**: Verify `ELICFeatureExtractor` instantiates from a downloaded checkpoint and outputs correctly shaped $y \in \mathbb{R}^{B \times 320 \times 16 \times 16}$ and $\hat{y} \in \mathbb{R}^{B \times 320 \times 16 \times 16}$ on a dummy image tensor.
*   **Verification Command**: `python -m pytest tests/test_extractor.py`

#### Phase 2: Probe Models & Multi-Head Container
*   **Goal**: Implement `LinearProbe`, `PreActBlock`, `LatentHead`, and `MultiHeadClassifierContainer`.
*   **Checkpoint**: Verify forward pass across all 36 heads with dummy tensor $(B, 320, 16, 16)$, verifying exact gradient computation and loss aggregation.
*   **Verification Command**: `python -m pytest tests/test_models.py`

#### Phase 3: Dataset Pipeline & Tier-1 HDF5 Cacher
*   **Goal**: Implement `caltech256.py`, `hdf5_dataset.py`, and `cache_latents.py`.
*   **Checkpoint**: Run mock caching on 10 images; verify HDF5 keys, `float16` dtypes, chunking, and SWMR read compatibility.
*   **Verification Command**: `python -m pytest tests/test_caching.py`

#### Phase 4: Tier-2 Training Loop & CI/CD Integration
*   **Goal**: Implement `train_probe.py`, `evaluate.py`, `ci/kaggle_runner.py`, and GitHub Actions workflows.
*   **Checkpoint**: Execute end-to-end dry-run (`train_probe.py --dry-run`); validate output CSV schema and Pareto curve generation.
*   **Verification Command**: `python -m benchmarks.train_probe --dry-run`

--------------------------------------------------------------------------------

## Verification & Test Plan

### Automated Test Suites
1.  **`tests/test_slicing.py`**:
    *   Verifies channel slice boundaries match exact specification: S0 (`0:16`), S1 (`16:32`), S2 (`32:64`), S3 (`64:128`), S4 (`128:320`), C1 (`0:32`), C2 (`0:64`), C3 (`0:128`), C4 (`0:320`).
    *   Confirms total channels in C4 equals exactly 320.
2.  **`tests/test_latent_head.py`**:
    *   Tests `LatentHead` with variable input channels ($C_{\text{in}} \in \{16, 32, 64, 128, 320\}$), verifying output logits shape is strictly $(B, 257)$.
    *   Verifies `PreActBlock` shortcut with `stride=2` correctly downsamples spatial dimensions from $16 \times 16$ to $8 \times 8$ without dimension mismatch.
3.  **`tests/test_augmentation.py`**:
    *   Verifies stratified split produces zero overlap between train and test image paths.
    *   Verifies classes with $< 150$ samples are augmented to exactly 150, while test images remain 100% clean and unaugmented.
4.  **`tests/test_hdf5_dataset.py`**:
    *   Validates multi-worker DataLoader reading from HDF5 under `swmr=True`.
    *   Verifies `float16` tensors are automatically cast to `float32` upon `__getitem__` retrieval.
5.  **`tests/test_multi_head_container.py`**:
    *   Verifies all 36 heads receive their respective slice views, compute cross-entropy losses, and propagate gradients into isolated parameter tensors.

--------------------------------------------------------------------------------

*Drafted with EGM*