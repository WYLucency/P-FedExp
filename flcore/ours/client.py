import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from utils.basic_utils import check_path
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
import numpy as np
import gensim.downloader
import gensim.downloader as api
from model.GFT_encoder import Encoder, InnerProductDecoder
from model.GFT_ft_model import TaskModel, TaskModelWithoutVQ
from model.GFT_pt_model import PretrainModel
from model.GFT_moe import EuclideanExperts, EuclideanGating
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import mask_feature, dropout_adj
from torch_geometric.data import Data
import copy
from torch_geometric.utils import subgraph
from utils.basic_utils import accuracy, index_to_mask
from utils.basic_utils import EarlyStopping, Logger
from utils.basic_utils import get_loader, seed_everything, get_preprocess,sample_proto_instances_for_graph
from task.node_cls import ft_node, eval_node
from task.link_pre import ft_link, eval_link
from task.graph_cls import ft_graph, eval_graph
from torch_geometric.loader import DataLoader
from model.GFT_ft_model import efficient_compute_class_prototypes
from tqdm import tqdm
from utils.partition_utils import merge_raw
from torch_geometric.nn.inits import glorot
# from model.non_param import LapSmoConv, LapSmoEncoder
import time


class Client:
    
    def __init__(self, args, client_id, data_tag, device, finetune_params=None, mode="pretrain"):
        self.args = args
        self.client_id = client_id
        self.mode = mode
        
        # 图分类数据集（包括 TUDataset）在 finetune 模式下不能调用 .to(device)，因为它们是 Dataset 切片
        if data_tag.name.lower() not in ["chemhiv", "chemblpre", "chempcba", "proteins", "imdb-binary"]:
            self.data_tag = data_tag.to(device)
            # self.data_tag = data_tag # for empirical
        else:
            self.data_tag = data_tag
            if mode == "pretrain":
                self.data_tag = self.data_tag.to(device)
                # self.data_tag = self.data_tag # for empirical

        self.device = device
        
        
        self.dim = 768
          
        # encoder-decoder
        # encoder-decoder
        # Initialize Experts instead of single Encoder
        self.num_experts = args.num_experts # Default number of experts
        self.moe_encoder = EuclideanExperts(
            num_experts=self.num_experts,
            input_dim=self.dim,
            hidden_dim=self.dim,
            activation=nn.ReLU,
            num_layers=2,
            backbone="mysage",
            normalize="batch",
            dropout=0.15
            )

        self.gating = EuclideanGating(
            input_dim=self.dim,
            hidden_dim=self.dim,
            num_experts=self.num_experts,
            dropout=0.15
        )

        self.feat_recon_decoder = nn.Linear(self.dim, self.dim)
        self.topo_recon_decoder = InnerProductDecoder(hidden_dim=self.dim, output_dim=self.dim)
        self.topo_sem_recon_decoder = nn.Linear(self.dim * 2, self.dim)
        
        
        # pretrain model
        self.pretrain_model = PretrainModel(
            moe_encoder=self.moe_encoder, gating=self.gating,
            feat_recon_decoder=self.feat_recon_decoder,
            topo_recon_decoder=self.topo_recon_decoder,
            topo_sem_recon_decoder=self.topo_sem_recon_decoder,
        ).to(device)

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
        if finetune_params is None:
            self.lambda_proto = 0
            self.lambda_act = 0
            self.num_instances_per_class = 0
            self.lambda_proto=0
            self.lambda_act=0
            self.num_instances_per_class=0
            self.trade_off=0
            self.finetune_batch_size=0
            self.finetune_lr = 0
            self.early_stop = 0
            self.finetune_epochs = 0
        else:
            self.lambda_proto=finetune_params["lambda_proto"]
            self.lambda_act=finetune_params["lambda_act"]
            self.num_instances_per_class=finetune_params["num_instances_per_class"]
            self.lambda_proto=finetune_params["lambda_proto"]
            self.lambda_act=finetune_params["lambda_act"]
            self.num_instances_per_class=finetune_params["num_instances_per_class"]
            self.trade_off=finetune_params["trade_off"]
            self.finetune_batch_size=finetune_params["batch_size"]
            self.finetune_lr = finetune_params["finetune_lr"]
            self.early_stop = finetune_params["early_stop"]
            self.finetune_epochs = finetune_params["finetune_epochs"]
        self.setting="standard"
        self.query_node_code_first=False
        
        self.max_batch = 20
        
    
    # def get_local_cluster_size(self):
    #     return self.vq._codebook.cluster_size
        
        
           
    def pretrain(self):
        time_start = time.time()
        print(f"[client {self.client_id}] pretraining...")
        self.pretrain_model.train()
        optimizer = AdamW(self.pretrain_model.parameters(), lr=self.pretrain_lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 + np.cos(epoch * np.pi / self.pretrain_epochs)) * 0.5)

        for i in range(1, self.pretrain_epochs + 1):
            all_batch_count = 0
                
            while all_batch_count < self.max_batch:
                
                batch_count = 0
                batch_size = self.pretrain_batch_size
                total_idx = torch.arange(0, self.data_tag.x.shape[0]).long()
                loader = NeighborLoader(self.data_tag, input_nodes=total_idx,
                                        num_neighbors=[10] * 2,
                                        batch_size=batch_size, shuffle=True) # shuffle 本意是防止一个领域的数据集集中在某一个 batch, 但是我们的设置下, 这个现象是必然的
                
                total_feat_recon_loss = 0
                total_topo_recon_loss = 0
                total_topo_sem_recon_loss = 0
                total_sem_recon_loss = 0
                total_commit_loss = 0
                total_loss = 0

                
                
                for batch_data in loader:
                    if all_batch_count == self.max_batch:
                        break
                    data_x_is_idx = batch_data.x.size(0) != batch_data.node_text_feat.size(0)

                    if data_x_is_idx:
                        x = batch_data.node_text_feat[batch_data.x].to(self.device)
                    else:
                        x = batch_data.node_text_feat.to(self.device)
                    
                    edge_index = batch_data.edge_index.to(self.device)
                    edge_attr = batch_data.edge_text_feat[batch_data.xe].to(self.device)
                    
                    
                    graph = [x, edge_index, edge_attr]

                    aug_x, _ = mask_feature(x, p=self.feat_p)
                    aug_edge_index, aug_edge_attr = dropout_adj(
                        edge_index, edge_attr, p=self.edge_p, force_undirected=True, num_nodes=x.size(0)
                    )
                    aug_graph = [aug_x, aug_edge_index, aug_edge_attr]
                    #OFA要做的
                    z, quantize, indices, losses = self.pretrain_model(
                        aug_graph, graph, self.topo_recon_ratio, bs=batch_data.batch_size, no_codebook=False
                    )

                    feat_recon_loss = self.feat_lambda * losses['feat_recon_loss']
                    topo_recon_loss = self.topo_lambda * losses['topo_recon_loss']
                    topo_sem_recon_loss = self.topo_sem_lambda * losses['topo_sem_recon_loss']
                    sem_recon_loss = self.sem_lambda * losses['sem_recon_loss']
                    commit_loss = losses['commit_loss']
                    loss = feat_recon_loss + topo_recon_loss + topo_sem_recon_loss + sem_recon_loss + commit_loss

                    optimizer.zero_grad()
                    loss.backward()
                    
                    nn.utils.clip_grad_norm_(self.pretrain_model.parameters(), 1.0)
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    self.pretrain_model.ema_update_sem_encoder(decay=self.sem_encoder_decay)

                    losses = {
                        'losses/feat_recon_loss': feat_recon_loss.item(),
                        'losses/topo_recon_loss': topo_recon_loss.item(),
                        'losses/topo_sem_recon_loss': topo_sem_recon_loss.item(),
                        'losses/sem_recon_loss': sem_recon_loss.item(),
                        'losses/commit_loss': commit_loss.item(),
                        'losses/loss': loss.item(),
                    }
                    total_feat_recon_loss += feat_recon_loss.item()
                    total_topo_recon_loss += topo_recon_loss.item()
                    total_topo_sem_recon_loss += topo_sem_recon_loss.item()
                    total_sem_recon_loss += sem_recon_loss.item()
                    total_commit_loss += commit_loss.item()
                    total_loss += loss.item()
                    batch_count += 1
                    all_batch_count += 1
                    
                avg_feat_recon_loss = total_feat_recon_loss / batch_count
                avg_topo_recon_loss = total_topo_recon_loss / batch_count
                avg_topo_sem_recon_loss = total_topo_sem_recon_loss / batch_count
                avg_sem_recon_loss = total_sem_recon_loss / batch_count
                avg_commit_loss = total_commit_loss / batch_count
                avg_loss = total_loss / batch_count

                print({
                    'avg_losses/feat_recon_loss': avg_feat_recon_loss,
                    'avg_losses/topo_recon_loss': avg_topo_recon_loss,
                    'avg_losses/topo_sem_recon_loss': avg_topo_sem_recon_loss,
                    'avg_losses/sem_recon_loss': avg_sem_recon_loss,
                    'avg_losses/commit_loss': avg_commit_loss,
                    'avg_losses/loss': avg_loss,
                })    
            

        time_end = time.time()

        print("-------------", time_end-time_start)
        self.pretrain_model.eval()
        


    def finetune(self, standard=1):
        num_tasks = self.data_tag.num_tasks
        task = self.data_tag.task
          
        train_loader = None
        val_loader = None
        test_loader = None
        subgraph_loader = None
        process = get_preprocess(task)
        dataset = self.data_tag
        dataset = process(dataset)


        labels = self.data_tag.y

        
        num_classes = num_tasks if task == "graph_cls" else self.data_tag.num_global_classes
            
        split = {"train": dataset.train_mask, "valid": dataset.val_mask, "test": dataset.test_mask}
        
        
        self.logger = Logger()
        for idx in range(standard): # debug: only 3 split
            seed_everything(idx)
            if self.setting == "standard":
                split = split
            # elif self.setting in ["few_shot", "zero_shot", "in_context"]:
            #     if task in ["node", "link"]:
            #         split = get_split(split, labels, params)
            #     elif task == "graph":
            #         split = get_split_graph(split, labels, params)
            else:
                raise ValueError("Invalid Setting")
            # OFA要做的
            task_model = TaskModel(
                moe_encoder=copy.deepcopy(self.moe_encoder),
                gating=copy.deepcopy(self.gating),
                prompt=None,
                num_classes=num_classes,
                separate_decoder_for_each_head=self.separate_decoder_for_each_head,
                use_z_in_predict=self.use_z_in_predict,
                use_cosine_sim=self.use_cosine_sim,
                lambda_proto=self.lambda_proto,
                lambda_act=self.lambda_act,
                trade_off=self.trade_off,
                num_instances_per_class=self.num_instances_per_class,
                ).to(self.device)

            opt_params = task_model.parameters()
            task_opt = AdamW(opt_params, lr=self.finetune_lr)
            stopper = EarlyStopping(patience=self.early_stop)

            # 【修复显存问题】对于大数据集（如reddit），即使batch_size为0也强制使用mini_batch模式
            # 判断是否需要强制使用mini_batch：检查数据集大小（节点数或边数）
            force_mini_batch = False
            if task in ["node_cls", "link_pre"]:
                # 对于节点分类任务，检查节点数和边数
                num_nodes = dataset.x.shape[0] if hasattr(dataset, 'x') else 0
                num_edges = dataset.edge_index.shape[1] if hasattr(dataset, 'edge_index') else 0
                # 如果节点数超过100万或边数超过5000万，强制使用mini_batch
                if num_nodes > 1000000 or num_edges > 50000000:
                    force_mini_batch = True
                    if self.finetune_batch_size == 0:
                        # 自动设置合适的batch_size
                        effective_batch_size = 1024 if num_nodes < 5000000 else 512
                        print(f"[显存优化] 数据集 {self.data_tag.name} 较大 (节点数: {num_nodes}, 边数: {num_edges})，强制使用mini_batch模式，batch_size: {effective_batch_size}")
                    else:
                        effective_batch_size = self.finetune_batch_size
                else:
                    effective_batch_size = self.finetune_batch_size
            else:
                effective_batch_size = self.finetune_batch_size

            if (self.finetune_batch_size != 0 or force_mini_batch) and task in ["node_cls", "link_pre"]:
                train_loader, subgraph_loader = get_loader(dataset, split, labels, task, effective_batch_size)
            elif self.finetune_batch_size != 0 and task == "graph_cls":
                train_loader, val_loader, test_loader = get_loader(dataset, split, labels, task, self.finetune_batch_size)
            finetune = get_ft(task)
            evaluate = get_eval(task)
        
            pbar = tqdm(range(self.finetune_epochs), desc=f"Finetuning - Dataset {self.data_tag.name} - Standard {idx} - {self.data_tag.task}")
                    
            for epoch in pbar:
                loss = finetune(
                    model=task_model,
                    dataset=dataset if task in ["node_cls", "link_pre"] else dataset,
                    loader=train_loader,
                    optimizer=task_opt,
                    split=split,
                    labels=labels,
                    num_classes=num_classes,
                    no_proto_clf=self.no_proto_clf,
                    no_lin_clf=self.no_lin_clf,
                    use_z_in_predict=self.use_z_in_predict,
                    query_node_code_first=self.query_node_code_first,
                    lambda_proto=self.lambda_proto,
                    lambda_act=self.lambda_act,
                    num_instances_per_class=self.num_instances_per_class,
                    num_neighbors=[30] * 2,
                )

                result = evaluate(
                    model=task_model,
                    dataset= dataset if task in ["node_cls", "link_pre"] else dataset,
                    loader= subgraph_loader if task in ["node_cls", "link_pre"] else [train_loader, val_loader, test_loader],
                    split=split,
                    labels=labels,
                    num_classes=num_classes,
                    no_proto_clf=self.no_proto_clf,
                    no_lin_clf=self.no_lin_clf,
                    use_z_in_predict=self.use_z_in_predict,
                    query_node_code_first=self.query_node_code_first,
                    num_instances_per_class=self.num_instances_per_class,
                    task=task,
                    num_neighbors=[-1] * 2,
                )

                is_stop = stopper(result)
                self.logger.log(idx, epoch, loss, result)
                if is_stop:
                    print("Early Stopping at Epoch:", epoch)
                    break
                # if epoch%50==0:
                    # print("Epoch:", epoch)
            single_best = self.logger.get_single_best(idx)
        best = self.logger.get_best()
        print({
            "final/train": "{:.2f} ± {:.2f}".format(best['train']['mean'], best['train']['std']),
            "final/val": "{:.2f} ± {:.2f}".format(best['val']['mean'], best['val']['std']),
            "final/test": "{:.2f} ± {:.2f}".format(best['test']['mean'], best['test']['std']),
            "final/train_mean": best['train']['mean'],
            "final/val_mean": best['val']['mean'],
            "final/test_mean": best['test']['mean'],
            "final/train_std": best['train']['std'],
            "final/val_std": best['val']['std'],
            "final/test_std": best['test']['std'],
        })
    
        
        

    def get_pretrain_model(self):
        local_message = {
            "num_samples": 1,
            "weight": list(self.pretrain_model.named_parameters())
        }

        return local_message
    # 客户端在接收全局模型时，显式地跳过了码本 (codebook) 的覆盖。
    # 这意味着码本的更新遵循一套独立的逻辑（在 Server 端处理），或者客户端希望保留本地学到的离散特征分布。
    def set_pretrain_model(self, global_message): 
        with torch.no_grad():
            for (local_name, local_param), (_, global_param) in zip(self.pretrain_model.named_parameters(), global_message):
                local_param.data.copy_(global_param.data)




def get_ft(c_task):
    task = c_task

    if task == "node_cls":
        return ft_node
    elif task == "link_pre":
        return ft_link
    elif task == "graph_cls":
        return ft_graph
    else:
        raise ValueError("Invalid Task")            
                
def get_eval(c_task):
    task = c_task

    if task == "node_cls":
        return eval_node
    elif task == "link_pre":
        return eval_link
    elif task == "graph_cls":
        return eval_graph
    else:
        raise ValueError("Invalid Task")            