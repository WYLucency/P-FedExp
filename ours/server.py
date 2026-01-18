import time
import torch
from model.GFT_encoder import Encoder, InnerProductDecoder
from model.GFT_ft_model import TaskModel
from model.GFT_pt_model import PretrainModel
from model.GFT_vq import VectorQuantize
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import defaultdict
import copy
from utils.basic_utils import construct_graph
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.optim import Adam, AdamW
from scipy.stats import wasserstein_distance  # 计算一维 EMD
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import mask_feature, dropout_adj

class Server:
    
    def __init__(self, args, device, kmeans_init=False):
        self.per_global_message = None
        self.args = args
        self.device = device
        self.dim = 768
        
        self.encoder = Encoder(
            input_dim=self.dim,
            hidden_dim=self.dim,
            activation=nn.ReLU,
            num_layers=2,
            backbone="mysage",
            normalize="batch",
            dropout=0.15
            )

        self.vq = VectorQuantize(
            dim=self.dim,
            codebook_size=128,
            codebook_dim=self.dim,
            heads=4,
            separate_codebook_per_head=True,
            decay=0.8,
            commitment_weight=10,
            use_cosine_sim=True,  # Cosine Codebook Works, Euclidean Codebook Collapses
            orthogonal_reg_weight=1,
            orthogonal_reg_max_codes=32,
            orthogonal_reg_active_codes_only=False,
            kmeans_init=kmeans_init,
            ema_update=False,
        )

        self.feat_recon_decoder = nn.Linear(self.dim, self.dim)
        self.topo_recon_decoder = InnerProductDecoder(hidden_dim=self.dim, output_dim=self.dim)
        self.topo_sem_recon_decoder = nn.Linear(self.dim * 2, self.dim)


        # pretrain model
        self.global_model = PretrainModel(
            encoder=self.encoder, vq=self.vq,
            feat_recon_decoder=self.feat_recon_decoder,
            topo_recon_decoder=self.topo_recon_decoder,
            topo_sem_recon_decoder=self.topo_sem_recon_decoder,
        ).to(device)
        

        self.round_id = 0 
        
        # pretrain model params
        self.pretrain_epochs = 2
        self.pretrain_batch_size = 1024
        self.feat_p=0.2
        self.edge_p=0.2
        self.topo_recon_ratio=0.1
        self.feat_lambda=100
        self.topo_lambda=0.01
        self.topo_sem_lambda=100
        self.sem_lambda=1
        self.sem_encoder_decay=0.99
        self.pretrain_lr=1e-4
        self.separate_codebook_per_head=True
        self.separate_decoder_for_each_head=True
        self.use_cosine_sim=True
        self.use_z_in_predict=True
        self.no_lin_clf=False
        self.no_proto_clf=False
        
        


    def execute(self, local_message_dict, local_codebooks, get_gfm=False):
        clients_similarity = {}
        
        
 

        num_clients = len(local_codebooks)
        num_heads = local_codebooks[0].embed.data.shape[0]
        tokens_per_client = local_codebooks[0].embed.data.shape[1]

        for source_client in range(num_clients):
            clients_similarity[source_client] = {}
            for target_client in range(num_clients):
                clients_similarity[source_client][target_client] = 0


        for head in range(num_heads):
            combined_freq = []
            combined_tokens = []
            combined_idx = []
            for client_id in range(num_clients):
                combined_idx.append(torch.ones(tokens_per_client) * client_id)
                combined_freq.append(local_codebooks[client_id].cluster_size[head])
                combined_tokens.append(local_codebooks[client_id].embed.data[head])
            
            combined_freq = torch.hstack(combined_freq)
            combined_freq = combined_freq.view(-1,1).expand(combined_freq.shape[0], combined_freq.shape[0])
            combined_tokens = torch.vstack(combined_tokens)
            combined_idx = torch.hstack(combined_idx)
            combined_idx = combined_idx.view(-1,1).expand(combined_idx.shape[0], combined_idx.shape[0])
            same_client = (combined_idx == combined_idx.T).to(self.device)


            # Step 1: normalize each token to unit vector
            normalized = F.normalize(combined_tokens, p=2, dim=1)  # shape: [N, D]

            # Step 2: compute similarity via dot product
            similarity_matrix = torch.matmul(normalized, normalized.T) # (27*128) * (27*128)



            # client-wise similarity
            for source_client in range(num_clients):
                r_range = [source_client * tokens_per_client, (source_client+1) * tokens_per_client]
                for target_client in range(num_clients):
                    c_range = [target_client * tokens_per_client, (target_client+1) * tokens_per_client]

                    block = similarity_matrix[r_range[0]:r_range[1], c_range[0]:c_range[1]]

                    sim = block.max(dim=1)[0].mean()

                    clients_similarity[source_client][target_client] += sim


            directions = combined_freq.T <= combined_freq
            directions = directions & (~same_client) 

            diag_indices = torch.arange(directions.size(0), device=directions.device)
            no_valid_source = ~directions.any(dim=1)
            directions[diag_indices[no_valid_source], diag_indices[no_valid_source]] = True



            # Step 3: self mask & freq-aware mask
            masked_sim = similarity_matrix.masked_fill(~directions, float('-inf'))  # 用 -inf 屏蔽不蒸馏方向
            



            # softmax 归一化，按行（dim=1），仅对合法方向归一化
            normalized_weights = F.softmax(masked_sim * 2, dim=1)  # shape: [N, N]

            # Step 4: ema update
            

            updated_tokens = torch.matmul(normalized_weights, combined_tokens)

            for client_id in range(num_clients):
                start = client_id * tokens_per_client
                end = (client_id + 1) * tokens_per_client

    
                local_codebooks[client_id].embed.data[head] = local_codebooks[client_id].embed.data[head] * 0.8 + updated_tokens[start:end] * 0.2

            # coverage maximization
            print("-"*20)    


        # use client similarity
        self.personalized_glb_model_list = []

        for source_client in range(num_clients):
            normalized_agg_weights = []
            for target_client in range(num_clients):
                normalized_agg_weights.append(clients_similarity[source_client][target_client])
            normalized_agg_weights = torch.hstack(normalized_agg_weights).to(self.device)
            normalized_agg_weights = F.softmax(normalized_agg_weights * 2, dim=0)

            with torch.no_grad():
                for it, client_id in enumerate(local_message_dict):
                    weight = normalized_agg_weights[it]
                    local_named_weights = dict(local_message_dict[client_id]['weight'])
                    for name, global_param in self.global_model.named_parameters():
                        if name == 'vq._codebook.embed':
                            continue
                        local_param = local_named_weights[name]
                        if it == 0:
                            global_param.data.copy_(weight * local_param.data)
                        else:
                            global_param.data += weight * local_param.data

                self.personalized_glb_model_list.append(copy.deepcopy(self.get_global_message()))

        
        # gfm
        if get_gfm:
            # for tokens
            glb_client_weight = [0] * num_clients
            
            num_clients = len(local_codebooks)
            num_heads = local_codebooks[0].embed.data.shape[0]
            tokens_per_client = local_codebooks[0].embed.data.shape[1]


            for head in range(num_heads):
                combined_freq = []
                combined_tokens = []
                combined_idx = []
                for client_id in range(num_clients):
                    combined_idx.append(torch.ones(tokens_per_client) * client_id)
                    combined_freq.append(local_codebooks[client_id].cluster_size[head])
                    combined_tokens.append(local_codebooks[client_id].embed.data[head])
                
                combined_freq = torch.hstack(combined_freq)
                combined_freq = combined_freq.view(-1,1).expand(combined_freq.shape[0], combined_freq.shape[0])
                combined_tokens = torch.vstack(combined_tokens)
                combined_idx = torch.hstack(combined_idx)
                combined_idx = combined_idx.view(-1,1).expand(combined_idx.shape[0], combined_idx.shape[0])
                same_client = (combined_idx == combined_idx.T).to(self.device)


                # Step 1: normalize each token to unit vector
                normalized = F.normalize(combined_tokens, p=2, dim=1)  # shape: [N, D]

                # Step 2: token-wise similarity & glb client weight
                similarity_matrix = torch.matmul(normalized, normalized.T) # (27*128) * (27*128)
                for client_id in range(num_clients):
                    r_range = [tokens_per_client * client_id, tokens_per_client * (client_id+1)]
                    block = similarity_matrix[r_range[0]:r_range[1], :]
                    sim = block.mean()
                    glb_client_weight[client_id] += sim


            # for others
            glb_client_weight = torch.hstack(glb_client_weight).to(self.device)
            normalized_agg_weights = F.softmax(glb_client_weight * 2, dim=0)  # shape: [N, N]

            with torch.no_grad():
                for it, client_id in enumerate(local_message_dict):
                    weight = normalized_agg_weights[it]
                    local_named_weights = dict(local_message_dict[client_id]['weight'])
                    for name, global_param in self.global_model.named_parameters():
                        local_param = local_named_weights[name]
                        if it == 0:
                            global_param.data.copy_(weight * local_param.data)
                        else:
                            global_param.data += weight * local_param.data



    def execute_avg(self, local_message_dict):
        """
        FedAvg aggregation: simple weighted average of model parameters
        """
        num_clients = len(local_message_dict)
        
        with torch.no_grad():
            for it, client_id in enumerate(local_message_dict):
                weight = 1.0 / num_clients  # Equal weight for each client
                local_named_weights = dict(local_message_dict[client_id]['weight'])
                for name, global_param in self.global_model.named_parameters():
                    if name == 'vq._codebook.embed':
                        continue
                    local_param = local_named_weights[name]
                    if it == 0:
                        global_param.data.copy_(weight * local_param.data)
                    else:
                        global_param.data += weight * local_param.data
        
        # Set personalized_glb_model_list: for FedAvg, all clients use the same global model
        self.personalized_glb_model_list = []
        global_message = self.get_global_message()
        for _ in range(num_clients):
            self.personalized_glb_model_list.append(copy.deepcopy(global_message))

    def get_global_message(self):
        global_message = list(self.global_model.named_parameters())
        return global_message
        
