import random
import importlib

import tqdm

import torch
from torch.utils.data import Dataset
import pytorch3d.ops



if __name__ == "__main__":
    print("============ Patch Dataloader ============")
    pc_dataloader = importlib.import_module("pointcloud_dataloader")
    PCD = pc_dataloader.PointCloudDataset()
    print(PCD[0].shape)