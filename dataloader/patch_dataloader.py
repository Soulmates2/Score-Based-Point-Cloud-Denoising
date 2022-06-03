import random

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
        seed_idx = torch.randperm(noisy.shape[0])[:1]
        seed_points = noisy[seed_idx].unsqueeze(0)
        patch_noisy = knn_points(seed_points, noisy.unsqueeze(0), K=self.patch_size, return_nn=True)[2]
        patch_clean = knn_points(seed_points, clean_gt.unsqueeze(0), K=int(self.patch_ratio*self.patch_size), return_nn=True)[2]
        return patch_noisy[0], patch_clean[0]
    
    def __len__(self):
        return sum([len(dataset) for dataset in self.pc_dataset])*self.num_patches
    
    def __getitem__(self, idx):
        pc_dataset = random.choice(self.pc_dataset)
        pc_data = pc_dataset[idx % len(pc_dataset)]
        patch_noisy, patch_clean = self.make_patch_pointcloud(pc_data['noisy_pc'], pc_data['clean_pc'])
        data = {'noisy_pc': patch_noisy[0], 'clean_pc': patch_clean[0]}
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == "__main__":
    print("============ Patch Dataloader ============")
    