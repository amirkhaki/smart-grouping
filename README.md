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
- `[Network.py:get_scale_table]`: Helper calculating log-linear scale tables for Gaussian conditional entropy modeling.
- `[train.py:RateDistortionLoss]`: Loss module computing Lagrangian rate-distortion objective: `loss = lmbda * 255^2 * mse + bpp`.
- `[train.py:CustomDataParallel]`: Multi-GPU wrapper exposing inner model methods (`compress`, `decompress`, `update`).
- `[train.py:configure_optimizers]`: Configures distinct Adam optimizers for network parameters and auxiliary entropy bottleneck parameters.
- `[train.py:train_one_epoch]`: Runs one training epoch over image batches with gradient clipping.
- `[train.py:test_epoch]`: Evaluates validation loss, rate (bpp), and distortion (MSE).
- `[Inference.py:inference]`: Runs full compression and decompression on input image, computes PSNR and MS-SSIM, saves reconstruction, and logs metrics to CSV.
- `[Inference.py:inference_entropy_estimation]`: Estimates bitrate via negative log-likelihoods without arithmetic coding.
- `[Inference.py:eval_model]`: Iterates over image directory for full evaluation.
- `[updata.py:main]`: Updates CDF tables of trained models and serializes updated checkpoints.

## Subdirectories & System Boundaries
- benchmarks: [benchmarks](./benchmarks/README.md) - Benchmark execution suite for evaluating representations learned by ELIC models on Caltech-256 downstream classification, including latent caching to HDF5, multi-head probe training, and Rate-Accuracy Pareto visualization.
- ci: [ci](./ci/README.md) - Continuous integration and Kaggle script kernel bootstrap harness for remote GPU training and evaluation workflows.
- ELICUtilis: [ELICUtilis](./ELICUtilis/README.md) - Core utility subsystem grouping custom neural layers, dataset loaders, and model checkpoint management helpers.

## Global Invariants
- `[Network.py:TestModel]`: Input images must have 3 RGB channels with spatial dimensions divisible by downsampling factor 64 (or padded using `Inference.py` padding logic).
- `[Network.py:TestModel]`: Latent channel partition must conform to slice configuration `groups = [0, 16, 16, 32, 64, 192]` totaling `M=320` channels.

## Global Side Effects
- `[train.py:save_checkpoint]`: Checkpoint serialization to disk with deletion of stale checkpoints via `DelfileList`.
- `[train.py:main]`: Writes TensorBoard summary events.
- `[Inference.py:inference]`: Writes reconstructed images and appends rows to `result.csv`.
- `[updata.py:main]`: Overwrites or creates updated PyTorch checkpoint files.

## System Verification
- Run: `pytest`
