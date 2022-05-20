import os
import random
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from pytorch3d.ops import knn_points
# import wandb

from dataloader import *
from models.denoise import *
from models.util import *


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

def gradient_ascent_denoise(noisy_pc, model, patch_size=1000, denoise_knn=4, init_step_size=0.2, step_decay=0.95, num_steps=30):
    N = noisy_pc.size()[0] #(N,3)
    
    num_patches = int(3 * N / patch_size)
    patch_centers = farthest_point_sampling(noisy_pc, num_center_pts)
    noisy_patches = knn_points(patch_centers.unsqueeze(dim=0), noisy_pc.unsqueeze(dim=0), K=patch_size, return_nn=True)[0] #(M,P,3)
    
    with torch.no_grad():
        model.eval()
        model.feat_unit.eval()
        model.score_unit.eval()
        
        feat = model.feat_unit(noisy_patches)
        clean_patches = noisy_patches.clone()
        
        for i in range(num_steps):
            _, idx, nn = knn_points(noisy_patches, clean_patches, K=denoise_knn, return_nn=True) #idx: (M,P,knn)
            x = (nn - noisy_patches.unsqueeze(dim=2).repeat(1,1,denoise_knn,1)).reshape(-1,denoise_knn,3)
            z = feat.reshape(-1,feat.size()[-1])
            
            score = model.score_unit(x, z).reshape(noisy_patches.size()[0],-1,3) #(M*P,knn,3) -> (M,P*knn,3)
            gradients = torch.zeros_like(noisy_patches) #(M,P,3)
            gradients.scatter_add_(dim=1, index=idx.unsqueeze(-1).expand_as(score), src=score)
            
            step_size = init_step_size * (step_decay ** i)
            clean_patches += step_size * gradients
        
    return farthest_point_sampling(clean_patches.reshape(-1, 3), N)


def train_step(noisy_pc, clean_pc, model):
    noisy_pc = noisy_pc.to(device)
    clean_pc = clean_pc.to(device)
    
    model.train()
    loss = model.get_loss(noisy_pc, clean_pc)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), float("inf"))
    optimizer.step()

    return loss, grad_norm


def validation_step(dataset):
    clean_pc_list = []
    denoised_pc_list = []
    for i, test_data in enumerate(dataset):
        noisy_pc = test_data['noisy_pc'].to(device)
        denoised_pc = gradient_ascent_denoise(noisy_pc, model)
        denoised_pc_list.append(denoised_pc.unsqueeze(0))
        clean_pc_list.append(test_data['clean_pc'].unsqueeze(0).to(device))
    
    clean_pc_list = torch.stack(clean_pc_list)
    denoised_pc_list = torch.stack(denoised_pc_list)

    chamfer_distance = 1
    return chamfer_distance
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='PUNet', help='name of dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    
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
    device = 'cuda'

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
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    
    model = DenoiseNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    lowest_cd = float("inf")

    for epoch in range(1, args.epoch):
        for i, data in enumerate(train_loader):
            train_loss, grad_norm = train_step(data['noisy_pc'], data['clean_pc'], model)
            print(train_loss)
            if epoch % 2000 == 0 or epoch == args.epoch:
                cd = validation_step(test_data)
                print(cd)
                if cd < lowest_cd:
                    lowest_cd = cd