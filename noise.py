import sys
import math
import random

import torch
from torchvision.transforms import Compose


def normalize_pc(pc):
    # Normalize
    point_max = pc.max(dim=0)[0]
    point_min = pc.min(dim=0)[0]
    center = ((point_max + point_min)/2).unsqueeze(0)
    pc -= center
    # Scale
    scale = pc.pow(2).sum(dim=1).sqrt().max()
    pc /= scale
    return pc, center, scale


class UnitSphereNormalize(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data['clean_pc'], center, scale = normalize_pc(data['clean_pc'])
        data['center'] = center
        data['scale'] = scale
        return data


class RandomScale(object):
    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, data):
        scale = random.uniform(self.scale_min, self.scale_max)
        data['clean_pc'] = data['clean_pc'] * scale
        if 'noisy_pc' in data:
            data['noisy_pc'] = data['noisy_pc'] * scale
        return data


class RandomRotate(object):
    def __init__(self, degree_min=-180.0, degree_max=180.0, axis=0):
        self.degree_min = degree_min
        self.degree_max = degree_max
        self.axis = axis

    def __call__(self, data):
        degree = math.pi * random.uniform(self.degree_min, self.degree_max) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if self.axis == 0:
            matrix = torch.tensor([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        elif self.axis == 1:
            matrix = torch.tensor([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        elif self.axis == 2:
            matrix = torch.tensor([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        else:
            sys.exit("axis must be 0,1,2")

        data['clean_pc'] = torch.matmul(data['clean_pc'], matrix)
        if 'noisy_pc' in data:
            data['noisy_pc'] = torch.matmul(data['noisy_pc'], matrix)
        return data


class AddRandomNoise(object):
    def __init__(self, noise_std_min=0.005, noise_std_max=0.020):
        super().__init__()
        self.noise_std_min = noise_std_min
        self.noise_std_max = noise_std_max

    def __call__(self, data_dict):
        std = random.uniform(self.noise_std_min, self.noise_std_max)
        data_dict['noisy_pc'] = data_dict['clean_pc'] + torch.randn_like(data_dict['clean_pc']) * std
        data_dict['noise_std'] = std
        return data_dict


def train_transform(noise_std_min=0.005, noise_std_max=0.020, scale_min=0.8, scale_max=1.2, rotate=True):
    transforms = [
        UnitSphereNormalize(),
        AddRandomNoise(noise_std_min, noise_std_max),
        RandomScale(scale_min, scale_max)
    ]
    if rotate is True:
        transforms.append(RandomRotate(axis=0))
        transforms.append(RandomRotate(axis=1))
        transforms.append(RandomRotate(axis=2))
        
    return Compose(transforms)
