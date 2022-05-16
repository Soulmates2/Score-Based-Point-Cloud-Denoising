import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlockConv1d(nn.Module):
    def __init__(self, c_dim=63, input_size=128, hidden_size=128, output_size=128):
        super().__init__()
        self.c_dim = c_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.bn1 = nn.BatchNorm1d(self.input_size)
        self.act1 = nn.ReLU()
        self.conv1 = nn.Conv1d(self.input_size, self.hidden_size, 1)
        
        self.bn2 = nn.BatchNorm1d(self.hidden_size)
        self.act2 = nn.ReLU()
        self.conv2 = nn.Conv1d(self.hidden_size, self.output_size, 1)
        nn.init.zeros_(self.conv2.weight)

        self.conv3 = nn.Conv1d(self.c_dim, self.output_size, 1)
        
        self.conv4 = None if input_size == output_size else nn.Conv1d(input_size, output_size, 1, bias=False)

    
    def forward(self, x, condition):
        out = self.conv1(self.act1(self.bn1(x)))
        out1 = self.conv2(self.act2(self.bn2(out)))
        out2 = self.conv3(condition)
        shortcut = x if self.conv4 is None else self.conv4(x)

        return out1 + out2 + shortcut 



class ScoreNetwork(nn.Module):
    def __init__(self, point_dim=3, z_dim=60, grad_dim=3, hidden_dim=128, num_blocks=4):
        super().__init__()
        self.point_dim = point_dim
        self.z_dim = z_dim
        self.grad_dim = grad_dim
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv1d(point_dim + z_dim, hidden_dim, 1)

        self.layer_list = []
        for _ in range(num_blocks):
            self.layer_list.append(ResBlockConv1d(point_dim + z_dim, hidden_dim))
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_dim, grad_dim, 1)

        self.layers = nn.Sequential(*self.layer_list, self.bn1, self.act1, self.conv2)      
    
    def forward(self, x, z):
        B, N, D = x.size()
        point = x.transpose(2, 1)
        
        z_expand = z.unsqueeze(2).expand(-1, -1, N)
        condition = torch.cat([point, z_expand], dim=1)

        out = self.conv1(condition)
        out = self.layers(out, condition)
        out = out.transpose(2, 1)

        return out


if __name__ == "__main__":
    print("============ Score Network Module ============")
    