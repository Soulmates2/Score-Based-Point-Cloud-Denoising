import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseEdgeConv(nn.Module):
    def __init__(self, n_input, n_output):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        
    def forward(self, x):
        pass


if __name__ == "__main__":
    print("============ Feature Extraction Module ============")

