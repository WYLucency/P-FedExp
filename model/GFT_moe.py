import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GFT_encoder import Encoder
from torch_geometric.nn import GCNConv, global_mean_pool

class EuclideanExperts(nn.Module):
    def __init__(self, num_experts, input_dim, hidden_dim, activation, num_layers, backbone="mysage", normalize="batch", dropout=0.15):
        super(EuclideanExperts, self).__init__()
        self.experts = nn.ModuleList()
        self.num_experts = num_experts
        
        # In GraphMoRE, different experts had different curvatures. 
        # Here, we degenerate to Euclidean, so experts are identical in architecture 
        # but will learn different weights.
        for _ in range(num_experts):
            self.experts.append(
                Encoder(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    activation=activation,
                    num_layers=num_layers,
                    backbone=backbone,
                    normalize=normalize,
                    dropout=dropout
                )
            )

    def forward(self, x, edge_index, edge_attr=None):
        # Returns: [batch_size, hidden_dim, num_experts]
        expert_outputs = []
        for expert in self.experts:
            out = expert(x, edge_index, edge_attr)
            expert_outputs.append(out.unsqueeze(-1))
        
        return torch.cat(expert_outputs, dim=-1)

class EuclideanGating(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, dropout=0.15):
        super(EuclideanGating, self).__init__()
        self.encoder1 = GCNConv(input_dim, hidden_dim)
        self.encoder2 = GCNConv(hidden_dim, hidden_dim)
        
        # We assume node-level gating for now, as PretrainModel works on node embeddings.
        # If graph-level gating is needed (as in GraphMoRE's example which used pooling), 
        # we can add it, but for node classification/pretraining, node-wise weights make sense.
        # GraphMoRE used global_mean_pool for gating, suggesting graph-level experts?
        # Let's check Gating implementation in GraphMoRE again.
        # It used: encoder -> pooling -> classifier.
        # This implies it selected experts PER GRAPH (or subgraph).
        # However, FedBook seems to work on node embeddings for contrastive learning.
        # To be safe, let's implement NODE-LEVEL gating for versatility, 
        # but if the input is a batch of graphs, we can optionally pool.
        
        # For this implementation, I will stick to node-level gating to allow fine-grained expert selection.
        self.classifier = nn.Linear(hidden_dim, num_experts)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.encoder1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.encoder2(x, edge_index)
        x = F.relu(x)
        
        # Output unnormalized logits
        out = self.classifier(x)
        return out
