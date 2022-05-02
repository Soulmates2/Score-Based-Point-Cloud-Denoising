import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

class DenseEdgeConv(nn.Module):
    def __init__(self, n_input, n_output, knn, is_first_layer):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.knn = knn
        self.is_first_layer = is_first_layer
        
        if is_first_layer:
            self.layer1 = nn.Sequential(
                nn.Linear(n_input, n_output, bias=True),
                nn.ReLU()
            )
        else:
            self.layer1 = nn.Sequential(
                nn.Linear(3 * n_input, n_output, bias=True),
                nn.ReLU()
            )
        
        self.layer2 = nn.Sequential(
            nn.Linear(n_input + n_output, n_output, bias=True),
            nn.ReLU()
        )
        
        self.layer3 = nn.Linear(n_input + 2 * n_output, n_output, bias=True)
        
    def feature_knn(self, x):
        N, F = x.size()[1], x.size()[2]
        index = knn_points(x, x, K=self.knn+1)[1][:,:,1:]
        
        y = x.unsqueeze(dim=1).repeat(1, N, 1, 1)
        index = index.unsqueeze(dim=3).repeat(1, 1, 1, F)
        feat = torch.gather(y, dim=2, index=index)
        prev = x.unsqueeze(dim=2).expand_as(feat)
        
        if self.is_first_layer:
            return feat - prev
        else:
            return torch.cat([prev, feat, feat - prev], dim=-1)
        
    def forward(self, x):
        x1 = self.layer1(self.feature_knn(x))
        x2 = x.unsqueeze(dim=-2).repeat(1, 1, self.knn, 1)
        x = torch.cat([x1, x2], dim=-1)
        
        x1 = self.layer2(x)
        x2 = x
        x = torch.cat([x1, x2], dim=-1)
        
        x1 = self.layer3(x)
        x2 = x
        x = torch.cat([x1, x2], dim=-1)
        
        return torch.max(x, dim=-2)[0]
        
class FeatureExtraction(nn.Module):
    def __init__(self, input_size=3, feature_size=24, conv_output_size=12, n_layers=4, knn=16):
        super().__init__()
        self.input_size = input_size
        self.feature_size = feature_size
        self.n_layers = n_layers
        self.knn = knn
        
        self.layer_list = []       
        for i in range(n_layers):
            if i == 0:
                self.layer_list.append(nn.Linear(input_size, feature_size, bias=True))
                self.layer_list.append(DenseEdgeConv(feature_size, conv_output_size, knn, is_first_layer=True)
            else:
                self.layer_list.append(nn.Linear(feature_size + 3 * conv_output_size, feature_size, bias=True))
                self.layer_list.append(nn.ReLU())
                self.layer_list.append(DenseEdgeConv(feature_size, conv_output_size, knn, is_first_layer=False)
        self.layers = nn.Sequential(*self.layer_list)
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    print("============ Feature Extraction Module ============")

