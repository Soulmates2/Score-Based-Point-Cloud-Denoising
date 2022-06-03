import torch
import pytorch3d
from pytorch3d.loss import chamfer_distance, point_mesh_face_distance
from pytorch3d.ops import knn_points

from models.util import *

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
    # Normalize
    point_max = clean_pc.max(dim=-2, keepdim=True)[0]
    point_min = clean_pc.min(dim=-2, keepdim=True)[0]
    center = (point_max + point_min) / 2
    clean_pc -= center
    # Scale
    scale = clean_pc.pow(2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / 1.0
    gt = clean_pc / scale
    pred = (denoised_pc - center) / scale
    return chamfer_distance(pred, gt, batch_reduction='mean', point_reduction='mean')[0].item()


def compute_point_to_mesh(denoised_pc, verts, faces):
    # Normalize mesh
    verts = verts.unsqueeze(0)
    vertex_max = verts.max(dim=-2, keepdim=True)[0]
    vertex_min = verts.min(dim=-2, keepdim=True)[0]
    center = (vertex_max + vertex_min) / 2
    verts -= center
    # Scale
    scale = verts.pow(2).sum(dim=-1, keepdim=True).sqrt().max(dim=-2, keepdim=True)[0] / 1.0 
    verts /= scale
    verts = torch.squeeze(verts, dim=0)
    # Normalize pc
    denoised_pc.unsqueeze(0)
    denoised_pc = (denoised_pc - center)/scale
    denoised_pc = torch.squeeze(denoised_pc, dim=0)
    pc = pytorch3d.structures.Pointclouds([denoised_pc])
    mesh = pytorch3d.structures.Meshes([verts], [faces])
    return point_mesh_face_distance(mesh, pc).item()
