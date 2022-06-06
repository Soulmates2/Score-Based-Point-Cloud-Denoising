# Score-Based Point Cloud Denoising (ICCV'21)

[Paper] [https://arxiv.org/abs/2107.10981](https://arxiv.org/abs/2107.10981)

## This is re-implementation of Score-Based Point Cloud Denoising for 2022S KAIST 3DML project


## Installation

### Recommended Environment

The code has been tested in the following environment:

| Package                                                      | Version | Comment                                                      |
| ------------------------------------------------------------ | ------- | ------------------------------------------------------------ |
| PyTorch                                                      | 1.9.0   |                                                              |
| [point_cloud_utils](https://github.com/fwilliams/point-cloud-utils) | 0.18.0  | For evaluation only. It loads meshes to compute point-to-mesh distances. |
| [pytorch3d](https://github.com/facebookresearch/pytorch3d)   | 0.5.0   | For evaluation only. It computes point-to-mesh distances.    |
| [pytorch-cluster](https://github.com/rusty1s/pytorch_cluster) | 1.5.9   | We only use `fps` (farthest point sampling) to merge denoised patches. |

### Install via Conda (PyTorch 1.9.0 + CUDA 11.1)

```bash
conda env create -f env.yml
conda activate score-denoise
```

## Datasets

Download link: https://drive.google.com/file/d/1ZZ3EON8TTtwoRciT5ThcYU3sTtj9Kj7Z/view?usp=sharing

Please extract `score_dataset.zip` to `data` folder. It concludes PU-Net, PCNet, and noisy LiDAR dataset.


## Train

```bash
# basic training (supervised)
python train.py

# unsupervised training
python train.py --unsup True

# ablation study 2
python train.py --ablation2 True

```
Training time takes about 39~40 hours.

## Test

```bash
# PUNet 10K
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.01
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.02
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 2 --noise 0.03

# PUNet 50K
python test.py --dataset PUNet --resol 50000_poisson --denoise_iters 1 --noise 0.01
python test.py --dataset PUNet --resol 50000_poisson --denoise_iters 1 --noise 0.02
python test.py --dataset PUNet --resol 50000_poisson --denoise_iters 2 --noise 0.03

# PCNet 10K
python test.py --dataset PCNet --resol 10000_poisson --denoise_iters 1 --noise 0.01
python test.py --dataset PCNet --resol 10000_poisson --denoise_iters 1 --noise 0.02
python test.py --dataset PCNet --resol 10000_poisson --denoise_iters 2 --noise 0.03

# PCNet 50K
python test.py --dataset PUNet --resol 50000_poisson --denoise_iters 1 --noise 0.01
python test.py --dataset PUNet --resol 50000_poisson --denoise_iters 1 --noise 0.02
python test.py --dataset PUNet --resol 50000_poisson --denoise_iters 2 --noise 0.03
```

```bash
# Ablation study (1)
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.01 --ablation1 True
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.02 --ablation1 True
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.03 --ablation1 True

# Ablation study (1)+iters.
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.01 --ablation1 True
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.02 --ablation1 True
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 2 --noise 0.03 --ablation1 True

# Ablation study (2): use checkpoint trained by ablation2
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.01 --checkpoint ./checkpoints/ablation2_best.pt
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.02 --checkpoint ./checkpoints/ablation2_best.pt 
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 2 --noise 0.03 --checkpoint ./checkpoints/ablation2_best.pt

# Ablation study (3)
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.01 --ablation3 True
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 1 --noise 0.02 --ablation3 True
python test.py --dataset PUNet --resol 10000_poisson --denoise_iters 2 --noise 0.03 --ablation3 True
```

## Acknowledgement
We borrowed dataset, hyper-parameter setting and rendering code from author.

## Citation

```
@InProceedings{Luo_2021_ICCV,
    author    = {Luo, Shitong and Hu, Wei},
    title     = {Score-Based Point Cloud Denoising},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4583-4592}
}
```





