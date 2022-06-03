import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from .feature_extraction import FeatureExtraction
from .score_network import ScoreNetwork

class DenoiseNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.feat_unit = FeatureExtraction()
        self.score_unit = ScoreNetwork()
        self.num_pts = 128
        if not args.ablation2:
            self.knn_for_sample = 32
        else:
            self.knn_for_sample = 1
        if not args.ablation2_1:
            self.knn_for_score = 4
        else:
            self.knn_for_score = 1
        self.unsup_knn_for_score = 8
        print(self.knn_for_sample, self.knn_for_score)
    
    def sample_idx(self, n, r):
        L = list(range(n))
        random.shuffle(L)
        return L[:r]
    
    def get_loss(self, noisy_pc, clean_pc):
        B = noisy_pc.size()[0]
        N = noisy_pc.size()[1]
        M = clean_pc.size()[1]
        
        sampled_idx = self.sample_idx(N, self.num_pts)
        
        f_origin = noisy_pc[:,sampled_idx,:].unsqueeze(dim=2)
        noisy_feat = self.feat_unit(noisy_pc)[:,sampled_idx,:]
        f = knn_points(noisy_pc[:,sampled_idx,:], noisy_pc, K=self.knn_for_sample, return_nn=True)[2]
        
        x = (f - f_origin).reshape(-1, self.knn_for_sample, 3)
        z = noisy_feat.reshape(-1, noisy_feat.size()[2])
        estim_score = self.score_unit(x, z).reshape(B, self.num_pts, self.knn_for_sample, 3)
        
        merge_f = f.reshape(-1, self.knn_for_sample, 3)
        merge_f_origin = clean_pc.unsqueeze(dim=1).repeat(1, self.num_pts, 1, 1)
        nn = knn_points(merge_f, merge_f_origin.reshape(-1, M, 3), K=self.knn_for_score, return_nn=True)[2]
        
        ground_score = nn.reshape(-1, self.num_pts, nn.size()[1], nn.size()[2], 3)
        ground_score -= f.unsqueeze(dim=3)
        ground_score = torch.mean(ground_score, dim=3)
        
        loss = ((estim_score - ground_score).pow(2) * 50).sum(dim=-1).mean()
        return loss
        
    def get_unsupervised_loss(self, noisy_pc):
        B = noisy_pc.size()[0]
        N = noisy_pc.size()[1]
        
        sampled_idx = self.sample_idx(N, self.num_pts)
        
        f_origin = noisy_pc[:,sampled_idx,:].unsqueeze(dim=2)
        noisy_feat = self.feat_unit(noisy_pc)[:,sampled_idx,:]
        f = knn_points(noisy_pc[:,sampled_idx,:], noisy_pc, K=self.knn_for_sample, return_nn=True)[2]
        
        x = (f - f_origin).reshape(-1, self.knn_for_sample, 3)
        z = noisy_feat.reshape(-1, noisy_feat.size()[2])
        estim_score = self.score_unit(x, z).reshape(B, self.num_pts, self.knn_for_sample, 3)
        
        merge_f = f.reshape(-1, self.knn_for_sample, 3)
        merge_f_origin = noisy_pc.unsqueeze(dim=1).repeat(1, self.num_pts, 1, 1)
        nn = knn_points(merge_f, merge_f_origin.reshape(-1, N, 3), K=self.unsup_knn_for_score, return_nn=True)[2]
        
        ground_score = nn.reshape(-1, self.num_pts, nn.size()[1], nn.size()[2], 3)
        ground_score -= f.unsqueeze(dim=3)
        ground_score = torch.mean(ground_score, dim=3)
        
        loss = ((estim_score - ground_score).pow(2) * 50).sum(dim=-1).mean()
        return loss
        

if __name__ == "__main__":
    print("============ Denoising Module ============")