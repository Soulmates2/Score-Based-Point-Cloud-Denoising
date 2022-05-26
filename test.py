import os
import sys
import random
import argparse

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import point_cloud_utils
import pytorch3d
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance

from dataloader import *
from models.denoise import *
from models.util import *


def normalize_pc(pcl, center=None, scale=None):
    p_max = pcl.max(dim=0, keepdim=True)[0]
    p_min = pcl.min(dim=0, keepdim=True)[0]
    center = (p_max + p_min) / 2    # (1, 3)
    pcl = pcl - center
    
    scale = (pcl ** 2).sum(dim=1, keepdim=True).sqrt().max(dim=0, keepdim=True)[0]  # (1, 1)
    pcl = pcl / scale
    return pcl, center, scale


def load_xyz_files(dir):
    pc_dict = {}
    for fn in tqdm(os.listdir(dir), desc='xyz loading'):
        if os.path.splitext(fn)[1] != '.xyz':
            continue
        file_path = os.path.join(dir, fn)
        name = os.path.splitext(fn)[0]
        pc_dict[name] = torch.FloatTensor(np.loadtxt(file_path, dtype=np.float32))
    return pc_dict


def load_off_files(dir):
    mesh_dict = {}
    for fn in tqdm(os.listdir(dir), desc='mesh loading'):
        if os.path.splitext(fn)[1] != '.off':
            continue
        file_path = os.path.join(dir, fn)
        name = os.path.splitext(fn)[0]
        verts, faces = point_cloud_utils.load_mesh_vf(file_path)
        mesh_dict[name] = {'verts': torch.FloatTensor(verts), 'faces': torch.LongTensor(faces)}
    return mesh_dict



def gradient_ascent_denoise(noisy_pc, model, patch_size=1000, denoise_knn=4, init_step_size=0.2, step_decay=0.95, num_steps=30):
    N = noisy_pc.size()[0] #(N,3)
    
    num_patches = int(3 * N / patch_size)
    patch_centers = farthest_point_sampling(noisy_pc, num_patches)
    noisy_patches = knn_points(patch_centers.unsqueeze(dim=0), noisy_pc.unsqueeze(dim=0), K=patch_size, return_nn=True)[2][0] #(M,P,3)
    
    with torch.no_grad():
        model.eval()
        model.feat_unit.eval()
        model.score_unit.eval()
        
        feat = model.feat_unit(noisy_patches)
        iter_patches = noisy_patches.clone()
        # trace = [noisy_patches.clone().cpu()]
        
        for i in range(num_steps):
            _, idx, nn = knn_points(noisy_patches, iter_patches, K=denoise_knn, return_nn=True) #idx: (M,P,knn)
            x = (nn - noisy_patches.unsqueeze(dim=2).repeat(1,1,denoise_knn,1)).reshape(-1,denoise_knn,3)
            z = feat.reshape(-1,feat.size()[-1])
            
            score = model.score_unit(x, z).reshape(noisy_patches.size()[0],-1,3) #(M*P,knn,3) -> (M,P*knn,3)
            gradients = torch.zeros_like(noisy_patches) #(M,P,3)
            gradients.scatter_add_(dim=1, index=idx.reshape(idx.size()[0],-1,1).expand_as(score), src=score)
            
            step_size = init_step_size * (step_decay ** i)
            iter_patches += step_size * gradients
            # trace.append(iter_patches.clone().cpu())
        
    return farthest_point_sampling(iter_patches.reshape(-1, 3), N)


def compute_chamfer_distance(denoised_pc, clean_pc):
    # Normalize pc
    p_max = clean_pc.max(dim=-2, keepdim=True)[0]
    p_min = clean_pc.min(dim=-2, keepdim=True)[0]
    center = (p_max + p_min) / 2
    clean_pc = clean_pc - center
    # Scale
    scale = (clean_pc ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / 1.0  # (B, N, 1)
    gt = clean_pc / scale
    pred = (denoised_pc - center)/scale
    return chamfer_distance(pred, gt, batch_reduction='mean', point_reduction='mean')[0].item()


def compute_point_to_mesh(denoised_pc, verts, faces):
    # Normalize mesh
    verts = verts.unsqueeze(0)
    v_max = verts.max(dim=-2, keepdim=True)[0]
    v_min = verts.min(dim=-2, keepdim=True)[0]
    center = (v_max + v_min) / 2
    verts = verts - center
    # Scale
    scale = (verts ** 2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / 1.0  # (B, N, 1)
    verts = verts / scale
    verts = torch.squeeze(verts, dim=0)
    # Normalize pc
    denoised_pc.unsqueeze(0)
    denoised_pc = (denoised_pc - center)/scale
    denoised_pc = torch.squeeze(denoised_pc, dim=0)
    pc = pytorch3d.structures.Pointclouds([denoised_pc])
    mesh = pytorch3d.structures.Meshes([verts], [faces])
    return point_mesh_face_distance(mesh, pc).item()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./data', help='dataset path')
    parser.add_argument('--dataset', type=str, default='PUNet', help='name of dataset')
    parser.add_argument('--resol', type=str, default='10000_poisson', help='resolution of dataset')
    parser.add_argument('--noise', type=str, default='0.01', help='noise level of dataset')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best.pt')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    # Denoising
    parser.add_argument('--denoise_iters', type=int, default=1)
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

    save_dir = os.path.join('result', f'{args.dataset}_{args.resol}_{args.noise}')
    os.makedirs(save_dir, exist_ok=True)
    
    # load test data
    input_dir = os.path.join(args.root, 'examples', f'{args.dataset}_{args.resol}_{args.noise}')
    try:
        file_list = os.listdir(input_dir)
    except:
        file_list = os.listdir(input_dir)
        sys.exit("Test files don't exist")

    data_dict = {'file_name': [], 'noisy_pc': [], 'center': [], 'scale': []}

    # normalize data
    for fn in file_list:
        file_path = os.path.join(input_dir, fn)
        noisy_pc = torch.FloatTensor(np.loadtxt(file_path))
        normalized_pc, center, scale = normalize_pc(noisy_pc)
        data_dict['file_name'].append(fn[:-4])
        data_dict['noisy_pc'].append(normalized_pc)
        data_dict['center'].append(center)
        data_dict['scale'].append(scale)

    # load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DenoiseNet().to(device)
    model.load_state_dict(checkpoint['model'])

    # test
    for i in tqdm(range(len(file_list)), desc="save denoised pc"):
        noisy_pc = data_dict['noisy_pc'][i].to(device)
        with torch.no_grad():
            iter_pc = noisy_pc.clone()
            for _ in range(args.denoise_iters):
                # denoising
                iter_pc = gradient_ascent_denoise(iter_pc, model, denoise_knn=args.denoise_knn)
            denoised_pc = iter_pc.cpu()

            # denormalize point cloud
            denoised_pc = denoised_pc * data_dict['scale'][i] + data_dict['center'][i]
        
        # save denoised point cloud
        file_path = os.path.join(save_dir, data_dict['file_name'][i] + '.xyz')
        np.savetxt(file_path, denoised_pc.numpy(), fmt='%.8f')
        print(f"Save denoised file {data_dict['file_name'][i]}")
    
    # evaluate
    denoised_pc_dir = save_dir
    gt_pc_dir = os.path.join(args.root, args.dataset, 'pointclouds', 'test', args.resol)
    mesh_dir = os.path.join(args.root, args.dataset, 'meshes', 'test')

    denoised_pc_dict = load_xyz_files(denoised_pc_dir)
    gt_pc_dict = load_xyz_files(gt_pc_dir)
    mesh_dict = load_off_files(mesh_dir)
    metric_dict = {}
    cd_list = []
    p2m_list = []
    pc_name_list = list(denoised_pc_dict.keys())
    
    for name in tqdm(pc_name_list, desc="eval"):
        if name not in gt_pc_dict:
            print(f'{name}: Cannot validate without ground truth point cloud')
            continue
        pred_pc = denoised_pc_dict[name][:,:3].unsqueeze(0).to(device) # 1, N(resol), 3
        gt_pc = gt_pc_dict[name][:,:3].unsqueeze(0).to(device) # 1, N(resol), 3
        verts = mesh_dict[name]['verts'].to(device)
        faces = mesh_dict[name]['faces'].to(device)

        cd = compute_chamfer_distance(pred_pc, gt_pc)
        p2m = compute_point_to_mesh(pred_pc, verts, faces)

        metric_dict[name] = {'cd': cd, 'p2m': p2m}
        cd_list.append(cd)
        p2m_list.append(p2m)
    
    print(f'cd: {format(np.mean(cd_list), ".10f")}')
    print(f'p2m: {format(np.mean(p2m_list), ".10f")}')
    txt_file = os.path.join(save_dir, "eval_result.txt")
    with open(txt_file, 'w') as f:
        f.write(f'{args.dataset}_{args.resol}_{args.noise} evaluation result\n')
        f.write(f'cd: {format(np.mean(cd_list), ".10f")}\n')
        f.write(f'p2m: {format(np.mean(p2m_list), ".10f")}')
