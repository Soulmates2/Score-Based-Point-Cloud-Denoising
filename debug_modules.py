import os
import sys

from dataloader import *
from models import *
import models.feature_extraction
import models.score_network

from torch.utils.data import DataLoader


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
        PointCloudDataset(
            resolution=resol,
            transform=add_random_noise(noise_std_min=0.005, noise_std_max=0.02)
        ) for resol in ['10000_poisson', '30000_poisson', '50000_poisson']
    ],
)
print(patch_dataset[0]['noisy_pc'].shape, patch_dataset[0]['clean_pc'].shape)

patch_loader = DataLoader(patch_dataset, batch_size=32, num_workers=4, shuffle=True)

feature_extraction = models.feature_extraction.FeatureExtraction()
print(feature_extraction)

score_network = models.score_network.ScoreNetwork()
print(score_network)


for i, patch_dict in enumerate(patch_loader):
    feature = feature_extraction(patch_dict['noisy_pc'])
    print(f"extracted feature shape : {feature.shape}") # B, 1000, 60
    break

