import os
import sys
import random
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
# import wandb

from dataloader import *
from models.denoise import *


def normalize_pc(pcl, center=None, scale=None):
    p_max = pcl.max(dim=0, keepdim=True)[0]
    p_min = pcl.min(dim=0, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (1, 3)
    pcl = pcl - center
    
    scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
    pcl = pcl / scale
    return pcl, center, scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='PUNet', help='name of dataset')
    parser.add_argument('--resol', type=str, default='10000_poisson', help='resolution of dataset')
    parser.add_argument('--noise', type=str, default='0.01', help='noise level of dataset')
    parser.add_argument('--checkpoint', type=str, default='./pretrained/ckpt.pt', help='noise level of dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # Denoising
    parser.add_argument('--denoise_iters', type=int, default=1)
    parser.add_argument('--ld_step_size', type=float, default=None)
    parser.add_argument('--ld_step_decay', type=float, default=0.95)
    parser.add_argument('--ld_num_steps', type=int, default=30)
    parser.add_argument('--seed_k', type=int, default=3)
    parser.add_argument('--denoise_knn', type=int, default=4, help='ensembled score function')
    
    args = parser.parse_args()
    
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = 'cuda'

    save_dir = os.path.join('result', 'pointcloud')
    
    # load test data
    input_dir = os.path.join(args.root, 'examples', f'{args.dataset}_{args.resol}_{args.noise}')
    try:
        file_list = os.listdir(input_dir)
    except:
        file_list = os.listdir(input_dir)
        sys.exit("Test files don't exist")

    data_dict = {'file_name': [], 'noisy_pc': [], 'center': [], 'scale': []}
    num_data = 0

    # normalize data
    for fn in file_list:
        file_path = os.path.join(input_dir, fn)
        noisy_pc = torch.FloatTensor(np.loadtxt(file_path))
        normalized_pc, center, scale = normalize_pc(noisy_pc)
        data_dict['file_name'] += fn[:-4]
        data_dict['noisy_pc'] += normalized_pc
        data_dict['center'] += center
        data_dict['scale'] += scale
        num_data += 1

    # load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DenoiseNet(checkpoint['args']).to(device)
    model.load_state_dict(checkpoint['state_dict'])

    # test
    for i in tqdm(range(num_data)):
        noisy_pc = data_dict['noisy_pc'][i].to(device)
        with torch.no_grad():
            model.eval()
            intermediate_pc = noisy_pc
            for _ in range(args.denoise_iters):
                # denoising
                intermediate_pc = intermediate_pc
            denoised_pc = intermediate_pc.cpu()

            # denormalize point cloud
            denoised_pc = denoised_pc * data_dict['scale'][i] + data_dict['center'][i]
        
        # save denoised point cloud
        file_path = os.path.join(save_dir, data_dict['file_name'][i] + '.xyz')
        np.savetxt(file_path, denoised_pc.numpy(), fmt='%.8f')
        print(f"Save denoised file {data_dict['file_name'][i]}")
    
    # evaluate



        

    


    
    