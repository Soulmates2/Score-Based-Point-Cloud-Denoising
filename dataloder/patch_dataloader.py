import random
import importlib

import tqdm

import torch
from torch.utils.data import Dataset
from pytorch3d.ops import knn_points


class PatchDataset(Dataset):
    def __init__(self, pc_dataset, patch_ratio=1.2, patch_size=1000, num_patches=1000, transform=None):
        self.pc_dataset = pc_dataset
        self.patch_ratio = patch_ratio
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform
        
    def make_patch_pointcloud(self, noisy, clean_gt):
        seed_idx = torch.randperm(noisy.shape[0])[:self.num_patches]
        seed_points = noisy[seed_idx].unsqueeze(0)
        _, _, patch_noisy = knn_points(seed_points, noisy.unsqueeze(0), K=self.patch_size, return_nn=True)
        _, _, patch_clean = knn_points(seed_points, clean_gt.unsqueeze(0), K=int(self.patch_ratio*self.patch_size), return_nn=True)
        return patch_noisy[0], patch_clean[0]
    

    def __len__(self):
        return len(self.pc_dataset)*self.patch_ratio
    
    def __getitem__(self, idx):
        pc_dataset = random.choice(self.pc_dataset)
        pc_data = pc_dataset[idx % len(pc_dataset)]
        print("pc_data before patch:", pc_data['noisy_pc'].shape, pc_data['clean_pc'].shape)
        patch_noisy, patch_clean = self.make_patch_pointcloud(pc_data['noisy_pc'], pc_data['clean_pc'])
        data = {'noisy_pc': patch_noisy[0], 'clean_pc': patch_clean[0]}
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    print("============ Patch Dataloader ============")
    pc_dataloader = importlib.import_module("pointcloud_dataloader")
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
    
    patch_dataset = PatchDataset(
        pc_dataset=[
            pc_dataloader.PointCloudDataset(
                resolution=resol,
                transform=add_random_noise(noise_std_min=0.005, noise_std_max=0.02)
            ) for resol in ['10000_poisson', '30000_poisson', '50000_poisson']
        ],
    )
    print(patch_dataset[0]['noisy_pc'].shape, patch_dataset[0]['clean_pc'].shape)