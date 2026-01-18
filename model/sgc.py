import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import SGConv

# SGC Model
class SGC(nn.Module):
    def __init__(self, in_dim, out_dim, K=2):
        super(SGC, self).__init__()
        self.conv = SGConv(in_dim, out_dim, K=K)
    
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x
