import os
import random
import argparse

import wandb
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataloader import *
from models.denoise import *
from models.util import *
from noise import *
from eval_metric import *


def get_data_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def train_step(iter, model, unsup=False):
    batch = next(train_iter)
    noisy_pc = batch['noisy_pc'].to(device)
    clean_pc = batch['clean_pc'].to(device)
    
    model.train()
    if not unsup:
        loss = model.get_loss(noisy_pc, clean_pc)
    else:
        loss = model.get_unsupervised_loss(noisy_pc)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), float("inf"))
    optimizer.step()

    return loss, grad_norm


def validation_step(dataset):
    clean_pc_list = []
    denoised_pc_list = []
    for i, test_data in enumerate(tqdm(dataset, desc="val")):
        noisy_pc = test_data['noisy_pc'].to(device)
        denoised_pc = gradient_ascent_denoise(noisy_pc, model)
        denoised_pc_list.append(denoised_pc.unsqueeze(0))
        clean_pc = test_data['clean_pc'].to(device)
        clean_pc_list.append(clean_pc.unsqueeze(0))
        
    clean_pc_list = torch.cat(clean_pc_list, dim=0)
    denoised_pc_list = torch.cat(denoised_pc_list, dim=0)

    # if args.log is True:
    #     wandb.log({"point_scene": 
    #                     wandb.Object3D({
    #                         "type": "lidar/beta",
    #                         "points": denoised_pc_list[:4]
    #                     }
    #                 )})

    cd = compute_chamfer_distance(denoised_pc_list, clean_pc_list)
    return cd
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='PUNet', help='name of dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--epoch', type=int, default=1000000)
    parser.add_argument('--log', type=eval, default=True, choices=[True, False], help='logging train result')
    
    parser.add_argument('--unsup', type=eval, default=False, choices=[True, False], help='unsupervised learning')

    args = parser.parse_args()
    
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = f'cuda:{args.gpu}'

    if args.log is True:
        wandb.init(project='Score Denoising', config=args)
    
    ckpt_root = os.path.join(os.getcwd(), 'checkpoints')
    os.makedirs(ckpt_root, exist_ok=True)
    
    train_data = PatchDataset(
        pc_dataset=[
            PointCloudDataset(
                split='train',
                resolution=resol,
                transform=train_transform(noise_std_min=0.005, noise_std_max=0.020, scale_min=0.8, scale_max=1.2, rotate=True)
            ) for resol in ['10000_poisson', '30000_poisson', '50000_poisson']
        ],
    )
    test_data = PointCloudDataset(
        split='test',
        resolution='10000_poisson',
        transform=train_transform(noise_std_min=0.015, noise_std_max=0.015, scale_min=1, scale_max=1, rotate=False)
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    train_iter = get_data_iterator(train_loader)
    model = DenoiseNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    best_cd = float("inf")

    for epoch in range(1, args.epoch+1):
        train_loss, grad_norm = train_step(epoch, model, args.unsup)
        print(f'Epoch {epoch}/{args.epoch}: train loss = {train_loss}')
        if args.log is True:
            wandb.log({"Loss/Train": train_loss, "Grad/Train": grad_norm})
        if epoch % 2000 == 0 or epoch == args.epoch:
            cd = validation_step(test_data)
            print(f'Epoch {epoch}: chamfer distance = {cd}')
            ckpt = {'model':model.state_dict(), 
                    'optimizer':optimizer.state_dict(), 
                    'epoch':epoch,
                    'cd':cd}
            torch.save(ckpt, os.path.join(ckpt_root, 'last.pt'))
            if args.log is True:
                wandb.log({"CD/Test": cd})
            if cd < best_cd:
                best_cd = cd
                torch.save(ckpt, os.path.join(ckpt_root, 'best.pt'))
                print(f'Epoch {epoch}: {cd} model is saved')