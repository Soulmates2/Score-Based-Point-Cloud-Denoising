import os
import random
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
# import wandb

from dataloader import *
from models.denoise import *


class add_random_noise(object):
        def __init__(self, noise_std_min, noise_std_max):
            super().__init__()
            self.noise_std_min = noise_std_min
            self.noise_std_max = noise_std_max

        def __call__(self, data_dict):
            std = random.uniform(self.noise_std_min, self.noise_std_max)
            data_dict['noisy_pc'] = data_dict['clean_pc'] + torch.randn_like(data_dict['clean_pc']) * std
            data_dict['noise_std'] = std
            return data_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='PUNet', help='name of dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-4)
    
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--epoch', type=int, default=1000000)
    parser.add_argument('--log', type=eval, default=False, choices=[True, False], help='logging train result')

    args = parser.parse_args()
    
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.log is True:
        wandb.init(project="Score Denoising", config=args)
    
    train_data = PatchDataset(
        pc_dataset=[
            PointCloudDataset(
                split='train',
                resolution=resol,
                transform=add_random_noise(noise_std_min=0.005, noise_std_max=0.02)
            ) for resol in ['10000_poisson', '30000_poisson', '50000_poisson']
        ],
    )
    test_data = PointCloudDataset(
        split='test',
        resolution='10000_poisson',
        transform=add_random_noise(noise_std_min=0.005, noise_std_max=0.02)
    )
    