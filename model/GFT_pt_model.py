from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

EPS = 1e-15


class PretrainModel(nn.Module):
    def __init__(self, moe_encoder, gating, feat_recon_decoder, topo_recon_decoder, topo_sem_recon_decoder):
        super().__init__()

        self.moe_encoder = moe_encoder
        self.gating = gating
        self.feat_recon_decoder = feat_recon_decoder
        self.topo_recon_decoder = topo_recon_decoder
        self.topo_sem_recon_decoder = topo_sem_recon_decoder
        
        # We can keep sem_encoder for now if needed, or update it to be MoE too.
        # For simplicity, let's assume sem_encoder is a single encoder instance 
        # (maybe essentially one of the experts, or a separate shadow encoder).
        # Given the original code did `deepcopy(self.encoder)`, and now `self.moe_encoder` is a list,
        # we might want to just pick the first expert or create a fresh single encoder.
        # Let's create a fresh single encoder for semantic reconstruction target to avoid complexity.
        # Wait, `sem_encoder` was used for `sem_recon_loss`, providing a target representation.
        # If we use MoE, the target should probably also be consistent.
        # Let's disable sem_recon_loss logic related to `sem_encoder` updates for now 
        # or just make it a simple deepcopy of the FIRST expert to have *some* target.
        self.sem_encoder = deepcopy(self.moe_encoder.experts[0]) 
        self.sem_projector = nn.Linear(self.moe_encoder.experts[0].hidden_dim, self.moe_encoder.experts[0].hidden_dim)

    @property
    def get_encoder(self):
        return self.moe_encoder

    def save_encoder(self, path):
        torch.save(self.moe_encoder.state_dict(), path)
        
    def save_gating(self, path):
        torch.save(self.gating.state_dict(), path)

    def feat_recon(self, z):
        return self.feat_recon_decoder(z)

    def feat_recon_loss(self, z, x, bs=None):
        return F.mse_loss(self.feat_recon(z[:bs]), x[:bs])

    # Reconstructing tree structure, similar to graph reconstruction.
    def topo_recon_loss(self, z, pos_edge_index, neg_edge_index=None, ratio=1.0):

        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            # Randomly sample positive edges
            num_pos_edges = int(pos_edge_index.size(1) * ratio)
            num_pos_edges = max(num_pos_edges, 1)
            perm = torch.randperm(pos_edge_index.size(1))
            perm = perm[:num_pos_edges]
            pos_edge_index = pos_edge_index[:, perm]

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))

        pos_loss = -torch.log(self.topo_recon_decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()
        neg_loss = -torch.log(1 - self.topo_recon_decoder(z, neg_edge_index, sigmoid=True) + EPS).mean()

        return pos_loss + neg_loss

    # Reconstructing the edge feature between two nodes
    def topo_sem_recon_loss(self, z, edge_index, edge_attr, ratio=1.0):
        if ratio == 0.0:
            return torch.tensor(0.0, device=z.device)

        if ratio != 1.0:
            num_edges = int(edge_index.size(1) * ratio)
            num_edges = max(num_edges, 1)
            perm = torch.randperm(edge_index.size(1))
            perm = perm[:num_edges]
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        z = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        loss = F.mse_loss(self.topo_sem_recon_decoder(z), edge_attr)

        return loss

    # Reconstructing the tree representation
    def sem_recon_loss(self, g, z_moe, eta=1.0, bs=None):
        orig_x, orig_edge_index, orig_edge_attr = (
            g[0],
            g[1],
            g[2],
        )

        target_z = self.sem_encoder(orig_x, orig_edge_index, orig_edge_attr) # TODO: add detach?
        h = self.sem_projector(z_moe)

        target_z = F.normalize(target_z[:bs], dim=-1, p=2)  # N * D
        h = F.normalize(h[:bs], dim=-1, p=2)  # N * D

        loss = (1 - (target_z * h).sum(dim=-1)).pow_(eta)
        loss = loss.mean()

        return loss

    def ema_update_sem_encoder(self, decay=0.99):
        # Update sem_encoder with the weights of the first expert (simplified)
        for param_q, param_k in zip(self.moe_encoder.experts[0].parameters(), self.sem_encoder.parameters()):
            param_k.data = param_k.data * decay + param_q.data * (1 - decay)

    def encode(self, x, edge_index, edge_attr=None):
        # For inference, strictly speaking we should use the MoE mixing.
        # But some downstream tasks might expect a single encoder.
        # Let's perform the mixing.
        expert_outputs = self.moe_encoder(x, edge_index, edge_attr) # [N, hidden, num_experts]
        gate_logits = self.gating(x, edge_index) # [N, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(1) # [N, 1, num_experts]
        
        z = (expert_outputs * gate_weights).sum(dim=-1) # [N, hidden]
        return z

    def encode_graph(self, x, edge_index, edge_attr=None, batch=None, pool="mean"):
        z = self.encode(x, edge_index, edge_attr)
        if pool == "mean":
            z = global_mean_pool(z, batch)
        elif pool == "sum":
            z = global_add_pool(z, batch)
        elif pool == "max":
            z = global_max_pool(z, batch)
        return z

    def forward(self, aug_g, g, topo_recon_ratio=1.0, bs=None, no_codebook=False):
        # Note: no_codebook argument is kept for compatibility but ignored
        x, edge_index, edge_attr = aug_g[0], aug_g[1], aug_g[2]
        orig_x, orig_edge_index, orig_edge_attr = g[0], g[1], g[2]

        # MoE Forward Pass
        expert_outputs = self.moe_encoder(x, edge_index, edge_attr) # [N, hidden, num_experts]
        gate_logits = self.gating(x, edge_index) # [N, num_experts]
        gate_weights = F.softmax(gate_logits, dim=-1).unsqueeze(1) # [N, 1, num_experts]
        
        z = (expert_outputs * gate_weights).sum(dim=-1) # [N, hidden]
        
        # Losses
        feat_recon_loss = self.feat_recon_loss(z, orig_x, bs=bs)
        topo_recon_loss = self.topo_recon_loss(z, orig_edge_index, ratio=topo_recon_ratio)
        topo_sem_recon_loss = self.topo_sem_recon_loss(z, orig_edge_index, orig_edge_attr, ratio=topo_recon_ratio)
        sem_recon_loss = self.sem_recon_loss(g, z, eta=1.0, bs=bs)

        # Basic load balancing loss (standard for MoE)
        # Minimize entropy of mean gate weights to encourage diverse expert usage? 
        # Or Just simple L2 on logits? 
        # FedBook didn't have this, but MoE usually needs it. 
        # GraphMoRE used a distortion loss. 
        # Let's just return 0 for now to replicate "degenerate" behavior requested, 
        # unless balancing becomes an issue.
        commit_loss = torch.tensor(0.0, device=z.device)

        losses = {
            'feat_recon_loss': feat_recon_loss,
            'topo_recon_loss': topo_recon_loss,
            'topo_sem_recon_loss': topo_sem_recon_loss,
            'sem_recon_loss': sem_recon_loss,
            'commit_loss': commit_loss,
        }

        # Return format consistent with original: z, quantize, indices, losses
        # converting "quantize" and "indices" to None or dummy values since VQ is gone
        return z, z, None, losses

