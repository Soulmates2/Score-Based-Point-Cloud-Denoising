import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points
import random

from .feature_extraction import FeatureExtraction
from .score_network import ScoreNetwork

class DenoiseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.feat_unit = FeatureExtraction()
        self.score_unit = ScoreNetwork().to('cuda')
        self.num_train_pts = 128
        self.knn = 32
        self.knn_clean = 4
        self.sigma = 0.01
    
    def sample_idx(self, n, r):
        L = list(range(n))
        random.shuffle(L)
        return L[:r]
    
    def get_loss(self, noisy_pc, clean_pc):
        B = noisy_pc.size()[0]
        N = noisy_pc.size()[1]
        M = clean_pc.size()[1]
        
        sampled_idx = self.sample_idx(N, self.num_train_pts)
        
        noisy_feat = self.feat_unit(noisy_pc)[:,sampled_idx,:]
        f = knn_points(noisy_pc[:,sampled_idx,:], noisy_pc, K=self.knn, return_nn=True)[2]
        f_origin = noisy_pc[:,sampled_idx,:].unsqueeze(dim=2)
        
        estim_score = self.score_unit(
            (f - f_origin).reshape(-1, self.knn, 3),
            noisy_feat.reshape(-1, noisy_feat.size()[2])
        ).reshape(B, len(sampled_idx), self.knn, 3)
        
        merge_f = f.reshape(-1, self.knn, 3)
        merge_clean = clean_pc.unsqueeze(dim=1).repeat(1, self.num_train_pts, 1, 1).reshape(-1, M, 3)
        knn_clean = knn_points(merge_f, merge_clean, K=self.knn_clean, return_nn=True)[2]
        
        ground_score = f.unsqueeze(dim=3).repeat(1, 1, 1, self.knn_clean, 1) 
        ground_score -= knn_clean.reshape(-1, self.num_train_pts, self.knn, self.knn_clean, 3)
        ground_score = -1 * ground_score.mean(dim=3)
        
        loss = 0.5 * ((estim_score - ground_score) ** 2.0 * (1.0 / self.sigma)).sum(dim=-1).mean()
        return loss
        
        

if __name__ == "__main__":
    print("============ Denoising Module ============")