import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
import numpy as np
import random

from .feature_extraction import FeatureExtraction
from .score_network import ScoreNetwork


def get_random_indices(n, m):
        assert m < n
        return np.random.permutation(n)[:m]


class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feat_unit = FeatureExtraction()
        self.score_unit = ScoreNetwork()
        self.num_train_points = 128
        self.frame_knn = 32
        self.num_clean_nbs = 4
        self.dsm_sigma = 0.01
    
    
    def get_loss(self, pcl_noisy, pcl_clean):
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        # Feature extraction
        feat = self.feat_unit(pcl_noisy)  # (B, N, F)
        feat = feat[:,pnt_idx,:]  # (B, n, F)
        F = feat.size(-1)
        
        # Local frame construction
        _, _, frames = knn_points(pcl_noisy[:,pnt_idx,:], pcl_noisy, K=self.frame_knn, return_nn=True)  # (B, n, K, 3)
        frames_centered = frames - pcl_noisy[:,pnt_idx,:].unsqueeze(2)   # (B, n, K, 3)

        # Nearest clean points for each point in the local frame
        # print(frames.size(), frames.view(-1, self.frame_knn, d).size())
        _, _, clean_nbs = knn_points(
            frames.view(-1, self.frame_knn, d),    # (B*n, K, 3)
            pcl_clean.unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_clean, d),   # (B*n, M, 3)
            K=self.num_clean_nbs,
            return_nn=True,
        )   # (B*n, K, C, 3)
        clean_nbs = clean_nbs.view(B, len(pnt_idx), self.frame_knn, self.num_clean_nbs, d)  # (B, n, K, C, 3)

        # Noise vectors
        noise_vecs = frames.unsqueeze(dim=3) - clean_nbs  # (B, n, K, C, 3)
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 3)

        # Denoising score matching
        grad_pred = self.score_unit(
            x = frames_centered.view(-1, self.frame_knn, d),
            z = feat.view(-1, F),
        ).reshape(B, len(pnt_idx), self.frame_knn, d)   # (B, n, K, 3)
        grad_target = - 1 * noise_vecs   # (B, n, K, 3)

        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        
        return loss #, target, scores, noise_vecs

    def get_selfsupervised_loss(self, pcl_noisy):
        """
        Denoising score matching.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
        """
        B, N_noisy, d = pcl_noisy.size()
        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        # Feature extraction
        feat = self.feature_net(pcl_noisy)  # (B, N, F)
        feat = feat[:,pnt_idx,:]  # (B, n, F)
        F = feat.size(-1)
        
        # Local frame construction
        _, _, frames = knn_points(pcl_noisy[:,pnt_idx,:], pcl_noisy, K=self.frame_knn, return_nn=True)  # (B, n, K, 3)
        frames_centered = frames - pcl_noisy[:,pnt_idx,:].unsqueeze(2)   # (B, n, K, 3)

        # Nearest points for each point in the local frame
        # print(frames.size(), frames.view(-1, self.frame_knn, d).size())
        _, _, selfsup_nbs = knn_points(
            frames.view(-1, self.frame_knn, d),    # (B*n, K, 3)
            pcl_noisy.unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_noisy, d),   # (B*n, M, 3)
            K=self.num_selfsup_nbs,
            return_nn=True,
        )   # (B*n, K, C, 3)
        selfsup_nbs = selfsup_nbs.view(B, len(pnt_idx), self.frame_knn, self.num_selfsup_nbs, d)  # (B, n, K, C, 3)

        # Noise vectors
        noise_vecs = frames.unsqueeze(dim=3) - selfsup_nbs  # (B, n, K, C, 3)
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 3)

        # Denoising score matching
        grad_pred = self.score_net(
            x = frames_centered.view(-1, self.frame_knn, d),
            c = feat.view(-1, F),
        ).reshape(B, len(pnt_idx), self.frame_knn, d)   # (B, n, K, 3)
        grad_target = - 1 * noise_vecs   # (B, n, K, 3)

        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        return loss #, target, scores, noise_vecs
        
        

if __name__ == "__main__":
    print("============ Denoising Module ============")