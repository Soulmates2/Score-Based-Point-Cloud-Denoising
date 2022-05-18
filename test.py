import os
import random
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import wandb

from dataloader import *
from models.denoise import *


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--log', type=eval, default=True, choice=[True, False], help='log train result')

    args = parser.parse_args()
    
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if log is True:
        wandb.init(project="Score Denoising", config=args)
    
    
    