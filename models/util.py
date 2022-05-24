import torch
from torch_cluster import fps

def farthest_point_sampling(points, n):
    N = points.size()[0]
    index = fps(points, ratio=(n/N)+0.01, random_start=False)
    return points[index[:n], :]
    
    