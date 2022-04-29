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

## PU-Net Datasets

Download link: https://drive.google.com/drive/folders/1--MvLnP7dsBgBZiu46H0S32Y1eBa_j6P?usp=sharing

Please extract `data.zip` to `data` folder.


## PC-Net Datasets

Download link: https://nuage.lix.polytechnique.fr/index.php/s/9CrHe259zaJEMb8/download



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





