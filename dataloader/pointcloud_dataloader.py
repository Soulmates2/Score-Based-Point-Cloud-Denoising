import os
import random

import numpy as np
import tqdm

import torch
from torch.utils.data import Dataset
import pytorch3d.ops


class PointCloudDataset(Dataset):
    def __init__(self, root='../score-denoise/data', dataset="PUNet", split="train", resolution='10000_poisson', transform=None):
        self.data_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        self.transform = transform
        self.pc_data = []

        print(f"The number of {split} dataset : {len(os.listdir(self.data_dir))}")
        for fn in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, fn)
            data = np.loadtxt(file_path).astype(np.float32)
            self.pc_data.append(data)

    def __len__(self):
        return len(self.pc_data)

    def __getitem__(self, idx):
        data_dict = {'clean_pc': torch.FloatTensor(self.pc_data[idx])}
        if self.transform is not None:
            data_dict = self.transform(data_dict)
        return data_dict


if __name__ == "__main__":
    print("============ Point Cloud Dataloader ============")
    PCD = PointCloudDataset()
    print(len(PCD))
    print(PCD[0].shape) # 10000 (resl), 3
