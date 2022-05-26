import sys
import math
import random

import torch
from torchvision.transforms import Compose


def normalize_pc(pcl):
    p_max = pcl.max(dim=0, keepdim=True)[0]
    p_min = pcl.min(dim=0, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (1, 3)
    pcl = pcl - center
    
    scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
    pcl = pcl / scale
    return pcl, center, scale


class unit_sphere_normalize(object):
    def __init__(self):
        super().__init__()

    def __call__(self, data):
        data['clean_pc'], center, scale = normalize_pc(data['clean_pc'])
        data['center'] = center
        data['scale'] = scale
        return data


class random_scale(object):
    def __init__(self, scale_min, scale_max):
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, data):
        scale = random.uniform(self.scale_min, self.scale_max)
        data['clean_pc'] = data['clean_pc'] * scale
        if 'noisy_pc' in data:
            data['noisy_pc'] = data['noisy_pc'] * scale
        return data


class random_rotate(object):
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


def train_transform(noise_std_min=0.005, noise_std_max=0.020, scale_min=0.8, scale_max=1.2, rotate=True):
    transforms = [
        unit_sphere_normalize(),
        add_random_noise(noise_std_min, noise_std_max),
        random_scale(scale_min, scale_max)
    ]
    if rotate is True:
        transforms += [
            random_rotate(axis=0),
            random_rotate(axis=1),
            random_rotate(axis=2)
        ]
    return Compose(transforms)
