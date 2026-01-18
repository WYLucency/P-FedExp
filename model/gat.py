import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv



# GAT Model
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

