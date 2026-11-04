# ELIC: Efficient Learned Image Compression withUnevenly Grouped Space-Channel Contextual Adaptive Coding.

A PyTorch Implementation of "ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding."

> [!NOTE]
> This is not an official implementation code.

## System Overview
Implementation of the ELIC learned image compression architecture based on CompressAI. ELIC achieves state-of-the-art compression efficiency using unevenly grouped space-channel contextual adaptive coding (combining checkerboard masked convolutions and channel conditioning transforms) coupled with hyperprior entropy modeling.

More details can be found in the following paper:
```bibtex
@inproceedings{he2022elic,
  title={Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding},
  author={He, Dailan and Yang, Ziming and Peng, Weikun and Ma, Rui and Qin, Hongwei and Wang, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5718--5727},
  year={2022}
}
```

### Related Links
* CompressAI: https://github.com/InterDigitalInc/CompressAI
* ELIC-QVRF: https://github.com/VincentChandelier/ELIC-QVRF
* QVRF: https://github.com/bytedance/QRAF

### Available Checkpoints
| $\lambda$ | Link |
| --- | --- |
| 0.45 | [0.45](https://drive.google.com/file/d/1uuKQJiozcBfgGMJ8CfM6lrXOZWv6RUDN/view?usp=sharing) |
| 0.15 | [0.15](https://drive.google.com/file/d/1s544Uxv0gBY3WvKBcGNb3Fb22zfmd9PL/view?usp=sharing) |
| 0.032 | [0.032](https://drive.google.com/file/d/1Moody9IR8CuAGwLCZ_ZMTfZXT0ehQhqc/view?usp=sharing) |
| 0.016 | [0.0016](https://drive.google.com/file/d/1MWlYAmpHbWlGtG7MBBTPEew800grY5yC/view?usp=sharing) |
| 0.008 | [0.008](https://drive.google.com/file/d/1VNE7rx-rBFLnNFkz56Zc-cPr6xrBBJdL/view?usp=sharing) |
| 0.004 | [0.004](https://drive.google.com/file/d/1YGVJ9bpeEq0xfqka2xkaMzhDkeYFJi6q/view?usp=sharing) |

### Training Dataset
According to the paper, models are trained on the largest 8,000 images selected from the ImageNet dataset.
* Download: [ImageNet](http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar)
* Preprocessing and selection matches QVRF: https://github.com/bytedance/QRAF

### Environment & Installation
This code is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI):
```bash
pip3 install compressai==1.1.5
pip3 install thop
pip3 install ptflops
pip3 install timm
```

### Usage

#### Train
```bash
python3 train.py -d ./dataset --N 192 --M 320 -e 4000 -lr 1e-4 -n 8 --lambda 13e-3 --batch-size 16 --test-batch-size 16 --aux-learning-rate 1e-3 --patch-size 256 256 --cuda --save --seed 1926 --clip_max_norm 1.0
```
Fine-tuning (200 epochs):
```bash
python3 train.py -d ./dataset --N 192 --M 320 -e 4000 -lr 1e-4 -n 8 --lambda 13e-3 --batch-size 16 --test-batch-size 16 --aux-learning-rate 1e-3 --patch-size 256 256 --cuda --save --seed 1926 --clip_max_norm 1.0 --pretrained --checkpoint Pretrained4000epoch_checkpoint.pth.tar
```

#### Update Entropy Model
```bash
python3 -m updata.py checkpoint -n updatacheckpoint-name
```

#### Test / Evaluation
```bash
python Inference.py --dataset ./dataset/Kodak --output_path ELIC_0450_ft_3980_Plateau -p ./ELIC_0450_ft_3980_Plateau.pth.tar --patch 64
```

### Rate-Distortion Results
![](Kodak.png)

## Local Interface & Invariants
- `[Network.py:TestModel]`: Primary neural compression network implementing ELIC analysis/synthesis transforms, multi-slice latent channel conditioning, checkerboard context models, and hyperprior entropy coding.
- `[Network.py:ResidualBottleneckBlock]`: 3-stage residual bottleneck convolutional block with 1x1 down-projection, 3x3 convolution, and 1x1 up-projection.
- `[Network.py:Quantizer]`: Quantization module providing uniform additive noise (for training), Straight-Through Estimator (`ste`), and integer rounding.
- `[Network.py:get_scale_table]`: Generates an exponential scale table tensor from scale bounds (`SCALES_MIN=0.11`, `SCALES_MAX=256`, `SCALES_LEVELS=64`) for Gaussian conditional entropy modeling.
- `[Network.py:conv1x1]`: Helper creating a 2D 1x1 convolution layer with specified channels and stride.
- `[train.py:RateDistortionLoss]`: Loss module computing joint Lagrangian rate-distortion objective: `loss = lmbda * 255^2 * mse + bpp`.
- `[train.py:CustomDataParallel]`: Multi-GPU `nn.DataParallel` wrapper delegating attribute access and inner model methods (`compress`, `decompress`, `update`).
- `[train.py:configure_optimizers]`: Configures disjoint Adam optimizers for network parameters and auxiliary entropy bottleneck parameters.
- `[train.py:train_one_epoch]`: Runs one training epoch over image batches with backpropagation and gradient clipping.
- `[train.py:test_epoch]`: Evaluates validation loss, rate (bpp), and distortion (MSE) over evaluation dataloader.
- `[train.py:save_checkpoint]`: Saves serialized model, optimizer, and scheduler states to a checkpoint file.
- `[train.py:AverageMeter]`: Utility class tracking values and computing running arithmetic averages.
- `[train.py:main]`: Orchestrates training pipeline, dataset loading, optimization, TensorBoard logging, and checkpoint saving.
- `[Inference.py:read_image]`: Reads an image file from disk and converts it to a normalized PyTorch RGB tensor.
- `[Inference.py:inference]`: Runs full compression and decompression with arithmetic coding, measures PSNR and latency, saves reconstruction, and logs metrics to CSV.
- `[Inference.py:inference_entropy_estimation]`: Estimates bitrate via negative log-likelihoods without arithmetic coding.
- `[Inference.py:eval_model]`: Iterates over image directory for full evaluation, saving reconstructions and logging to `result.csv`.
- `[Inference.py:psnr]`: Calculates Peak Signal-to-Noise Ratio between two image tensors.
- `[Inference.py:collect_images]`: Recursively finds image file paths in a directory matching `IMG_EXTENSIONS`.
- `[updata.py:main]`: Loads checkpoint, runs `model.update(force=True)` to finalize CDF tables, and exports updated model checkpoint.

## Subdirectories & System Boundaries
- benchmarks: [benchmarks](./benchmarks/README.md) - Benchmark execution suite for evaluating representations learned by ELIC models on Caltech-256 downstream classification, including latent caching to HDF5, multi-head probe training, channel sparsity profiling, and Rate-Accuracy Pareto visualization.
- ci: [ci](./ci/README.md) - Automated execution harness and script runner for training and evaluating smart-grouping workloads on Kaggle GPU environments triggered via GitHub Actions workflows.
- ELICUtilis: [ELICUtilis](./ELICUtilis/README.md) - Core utility subsystem supporting the ELIC learned image compression implementation, grouping specialized neural network layers and transforms, dataset loading wrappers, and model checkpoint management utilities.

## Global Invariants
- `[Network.py:TestModel.decompress]`: Compressed strings container must be a 2-element list `[y_strings, z_strings]` corresponding to latent slice streams and hyperprior bitstream (`assert isinstance(strings, list) and len(strings) == 2`).
- `[Network.py:TestModel]`: Input images must have 3 RGB channels with spatial dimensions divisible by downsampling factor 64 (or padded using `Inference.py` padding logic).
- `[Network.py:TestModel]`: Latent channel partition must conform to slice configuration `groups = [0, 16, 16, 32, 64, 192]` totaling `M=320` channels.
- `[train.py:configure_optimizers]`: Main network optimizer and auxiliary quantiles optimizer must cover mutually disjoint and collectively exhaustive sets of trainable parameters (`assert len(inter_params) == 0`).
- `[train.py:parse_args]`: Dataset path `-d`/`--dataset` is required (`required=True`) for running the training script.
- `[Inference.py:read_image]`: Target image filepath must be an existing readable image file on disk (`assert os.path.isfile(filepath)`).
- `[Inference.py:setup_args]`: Checkpoint path `-p`/`--path` is required (`required=True`), and `-c`/`--entropy-coder` restricts values to `compressai.available_entropy_coders()`.
- `[updata.py:main]`: Input model checkpoint path must point to an existing valid serialized PyTorch checkpoint file.

## Global Side Effects
- `[train.py:save_checkpoint]`: Checkpoint serialization to disk with deletion of stale checkpoints via `DelfileList`.
- `[train.py:main]`: Writes TensorBoard summary event files to disk and sets global CUDA, cuDNN determinism, and random seed states.
- `[Inference.py:inference]`: Saves reconstructed images to disk with `torchvision.utils.save_image`, appends benchmark metrics to `result.csv`, and prints status to stdout.
- `[Inference.py:eval_model]`: Creates output directories, overwrites/appends to `result.csv`, and writes reconstructed image files.
- `[updata.py:main]`: In-place updates model CDF tables and writes updated PyTorch checkpoint files to disk.

## System Verification
- Run: `pytest`
