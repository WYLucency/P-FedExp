
import copy
import torch
import numpy as np
from torch_geometric.utils import to_scipy_sparse_matrix, to_networkx
from torch_geometric.data import Data
import sys
from sklearn.cluster import KMeans
import pymetis as metis
import torch_geometric.utils
from tqdm import tqdm
import torch_geometric
import os


def extract_floats(s):
    """
    Extracts and converts three floats separated by hyphens from a string and ensures their sum is 1.

    Args:
        s (str): A string containing three float numbers separated by hyphens (e.g., "0.6-0.3-0.1").

    Returns:
        tuple: A tuple of three floats (train, val, test) extracted from the string.

    Raises:
        AssertionError: If the sum of the three numbers does not equal 1.
    """
    from decimal import Decimal
    parts = s.split('-')
    train = float(parts[0])
    val = float(parts[1])
    test = float(parts[2])
    assert Decimal(parts[0]) + Decimal(parts[1]) + Decimal(parts[2]) == Decimal(1)
    return train, val, test

def idx_to_mask_tensor(idx_list, length):
    """
    Converts a list of indices to a tensor mask of a specified length.

    Args:
        idx_list (list[int]): List of indices that should be marked as 1 in the mask.
        length (int): Total length of the mask tensor.

    Returns:
        torch.Tensor: A binary mask tensor where positions corresponding to indices in idx_list are set to 1.
    """
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask



def mask_tensor_to_idx(tensor):
    """
    Converts a tensor mask to a list of indices where the tensor is non-zero.

    Args:
        tensor (torch.Tensor): A tensor containing binary values.

    Returns:
        list[int]: A list of indices corresponding to non-zero entries in the tensor.
    """
    result = tensor.nonzero().squeeze().tolist()
    if type(result) is not list:
        result = [result]
    return result
    
    
def local_subgraph_train_val_test_split_node(local_subgraph, split, num_classes, shuffle=True):
    num_nodes = local_subgraph.x.shape[0]
    

    train_, val_, test_ = extract_floats(split)
    
    train_mask = idx_to_mask_tensor([], num_nodes)
    val_mask = idx_to_mask_tensor([], num_nodes)
    test_mask = idx_to_mask_tensor([], num_nodes)
    
    
    for class_i in range(num_classes):
        class_i_node_mask = local_subgraph.y == class_i
        num_class_i_nodes = class_i_node_mask.sum()
        
        class_i_node_list = mask_tensor_to_idx(class_i_node_mask)
        if shuffle:
            np.random.shuffle(class_i_node_list)
        train_mask += idx_to_mask_tensor(class_i_node_list[:int(train_ * num_class_i_nodes)], num_nodes)
        val_mask += idx_to_mask_tensor(class_i_node_list[int(train_ * num_class_i_nodes) : int((train_+val_) * num_class_i_nodes)], num_nodes)
        test_mask += idx_to_mask_tensor(class_i_node_list[int((train_+val_) * num_class_i_nodes): min(num_class_i_nodes, int((train_+val_+test_) * num_class_i_nodes))], num_nodes)
    
    
    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    return train_mask, val_mask, test_mask




def local_subgraph_train_val_test_split_link(local_subgraph, split, num_classes, shuffle=True):
    num_edges = local_subgraph.edge_index.shape[1]
    

    train_, val_, test_ = extract_floats(split)
    
    train_mask = idx_to_mask_tensor([], num_edges)
    val_mask = idx_to_mask_tensor([], num_edges)
    test_mask = idx_to_mask_tensor([], num_edges)
    
    
    for class_i in range(num_classes):
        class_i_edge_mask = local_subgraph.y == class_i
        num_class_i_edges = class_i_edge_mask.sum()
        
        class_i_node_list = mask_tensor_to_idx(class_i_edge_mask)
        if shuffle:
            np.random.shuffle(class_i_node_list)
        train_mask += idx_to_mask_tensor(class_i_node_list[:int(train_ * num_class_i_edges)], num_edges)
        val_mask += idx_to_mask_tensor(class_i_node_list[int(train_ * num_class_i_edges) : int((train_+val_) * num_class_i_edges)], num_edges)
        test_mask += idx_to_mask_tensor(class_i_node_list[int((train_+val_) * num_class_i_edges): min(num_class_i_edges, int((train_+val_+test_) * num_class_i_edges))], num_edges)
    
    
    train_mask = train_mask.bool()
    val_mask = val_mask.bool()
    test_mask = test_mask.bool()
    return train_mask, val_mask, test_mask

# single graph partition

def get_subgraph_pyg_data_node(global_dataset, node_list):
    """
    e.g., Cora: Data(x=[2708], edge_index=[2, 10556], node_text_feat=[2708, 768], edge_text_feat=[1, 768], xe=[10556])
    x -> idx map to specific node_text_feat
    xe -> idx map to specific edge_text_feat
    edge_index -> topological structure
    node_text_feat -> node feature
    edge_text_feat -> edge feature
    """
    num_classes = torch.unique(global_dataset.y).shape[0]
    global_edge_index = global_dataset.edge_index
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = {}
    local_edge_list = []
    global_edge_map = []
    
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id[local_id] = global_id
        
    for edge_id in tqdm(range(global_edge_index.shape[1]), desc="Processing Edge Mapping"):
        src = global_edge_index[0, edge_id].item()
        tgt = global_edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))
            global_edge_map.append(edge_id)
            
            
    local_edge_index = torch.tensor(local_edge_list).T
    
    local_subgraph = Data(x=global_dataset.x[node_list], edge_index=local_edge_index, y=global_dataset.y[node_list])
    
    
    
    
    
    
    
    
    
    
    local_subgraph.global_map = local_id_to_global_id
    local_subgraph.xe = torch.vstack([global_dataset.data.xe[i] for i in global_edge_map]).squeeze()

        
    local_subgraph.node_text_feat = global_dataset.data.node_text_feat
    # 【修改点1】注释掉node_text，因为list类型会导致PyG的NeighborLoader报错
    # local_subgraph.node_text = global_dataset.node_text  # 【已移除】list类型不兼容NeighborLoader
    local_subgraph.edge_text_feat = global_dataset.data.edge_text_feat
    # 【修改点2】注释掉edge_text，同样的原因
    # local_subgraph.edge_text = global_dataset.edge_text  # 【已移除】list类型不兼容NeighborLoader
    
    local_subgraph.name = global_dataset.name
    local_subgraph.weight = global_dataset.weight

    local_subgraph.num_global_classes = num_classes
    local_subgraph.task = global_dataset.task
    local_subgraph.num_tasks = global_dataset.num_tasks
    
    # 【修改点3】在数据集列表中添加citeseer、photo、computers、reddit
    if local_subgraph.name.lower() in ["cora", "citeseer"]:
        local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = local_subgraph_train_val_test_split_node(local_subgraph, "0.2-0.4-0.4", num_classes=num_classes)
    elif local_subgraph.name.lower() in ["pubmed"]:
        local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = local_subgraph_train_val_test_split_node(local_subgraph, "0.6-0.2-0.2", num_classes=num_classes)
    elif local_subgraph.name.lower() in ["wikics"]:
        local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = local_subgraph_train_val_test_split_node(local_subgraph, "0.8-0.1-0.1", num_classes=num_classes)
    elif local_subgraph.name.lower() in ["arxiv"]:
        local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = local_subgraph_train_val_test_split_node(local_subgraph, "0.8-0.1-0.1", num_classes=num_classes)
    elif local_subgraph.name.lower() in ["photo", "computers", "reddit"]:
        local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = local_subgraph_train_val_test_split_node(local_subgraph, "0.2-0.4-0.4", num_classes=num_classes)
    
    return local_subgraph



def get_subgraph_pyg_data_link(global_dataset, node_list):
    """
    e.g., WN18RR
    Data(x=[40943], edge_index=[2, 93003], node_text_feat=[40943, 768], edge_text_feat=[1, 768], xe=[93003])
    
    """
    num_classes = torch.unique(global_dataset.y).shape[0]
    global_edge_index = global_dataset.edge_index
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = {}
    local_edge_list = []
    global_edge_map = []
    
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id[local_id] = global_id
        
    for edge_id in tqdm(range(global_edge_index.shape[1]), desc="Processing Edge Mapping"):
        src = global_edge_index[0, edge_id].item()
        tgt = global_edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))
            global_edge_map.append(edge_id)
            
            
    local_edge_index = torch.tensor(local_edge_list).T
    
    
    local_subgraph = Data(x=global_dataset.x[node_list], edge_index=local_edge_index, y=torch.vstack([global_dataset.y[edge_id] for edge_id in global_edge_map]).squeeze())
    local_subgraph.global_map = local_id_to_global_id
    local_subgraph.xe = torch.vstack([global_dataset.data.xe[i] for i in global_edge_map]).squeeze()
    
    
    local_subgraph.node_text_feat = global_dataset.data.node_text_feat
    # 【修改点4】注释掉node_text，list类型会导致PyG的NeighborLoader报错
    # local_subgraph.node_text = global_dataset.node_text  # 【已移除】
    local_subgraph.edge_text_feat = global_dataset.data.edge_text_feat
    # 【修改点5】注释掉edge_text，同样的原因
    # local_subgraph.edge_text = global_dataset.edge_text  # 【已移除】
    

    
    local_subgraph.name = global_dataset.name
    local_subgraph.weight = global_dataset.weight
    local_subgraph.num_global_classes = num_classes
    local_subgraph.task = global_dataset.task
    local_subgraph.num_tasks = global_dataset.num_tasks
    
    if local_subgraph.name.lower() in ["wn18rr", "fb15k237"]:
        local_subgraph.train_mask, local_subgraph.val_mask, local_subgraph.test_mask = local_subgraph_train_val_test_split_link(local_subgraph, "0.8-0.1-0.1", num_classes=num_classes)
    

    
    return local_subgraph






def single_graph_partition(data_tag, num_partitions, task="node_cls", root=None):
    dataset_dir = os.path.join(root, f"{data_tag.name.lower()}", f"tot_clients_{num_partitions}")
    os.makedirs(dataset_dir, exist_ok=True)
    filename_list = [os.path.join(dataset_dir, f"client_{i}.pt") for i in range(num_partitions)]
    
    all_partition_found = True
    for filename in filename_list:
        if not os.path.exists(filename):
            all_partition_found = False
            break
        
    local_data = []
    
       # METIS 是一种基于多级图划分的算法，其目标是最小化切边 (Edge Cut) 的数量，同时保持各分区的节点数量大致平衡。
    if not all_partition_found:
        print("Conducting subgraph-fl metis simulation...")
        graph_nx = to_networkx(data_tag[0], to_undirected=True)
        n_cuts, membership = metis.part_graph(num_partitions, graph_nx)
        
        client_indices = [None] * num_partitions
        for client_id in range(num_partitions):
            client_indices[client_id] = np.where(np.array(membership) == client_id)[0].tolist()
            
        
        
        for client_id, filename in enumerate(filename_list):
            if task == "node_cls":
                local_subgraph = get_subgraph_pyg_data_node(data_tag, client_indices[client_id])
            elif task == "link_pre":
                local_subgraph = get_subgraph_pyg_data_link(data_tag, client_indices[client_id])

            # save 
            torch.save(local_subgraph, filename)
        
        
    # use cache    
    for filename in filename_list:
        local_data.append(torch.load(filename))
        
        
    return local_data


def merge_raw(raw):
    merge_x = torch.hstack([i.x for i in raw])
    merge_y = torch.hstack([i.y for i in raw])
    merge_xe = torch.hstack([i.xe for i in raw])
    node_graph_id_map = []
    edge_graph_id_map = []
    
    # merge edge index
    ptr = 0
    edge_index_list = []
    pretrain_edge_index_list = []
    
    
    for id, i in enumerate(raw):
        edge_index_list.append(i.edge_index + ptr)
        pretrain_edge_index_list.append(i.pretrain_edge_index + ptr)
        ptr += i.x.shape[0]
        node_graph_id_map += [id] * i.x.shape[0]
        edge_graph_id_map += [id] * i.edge_index.shape[1]
        
    merge_edge_index = torch.hstack(edge_index_list)
    merge_pretrain_edge_index = torch.hstack(pretrain_edge_index_list)
    
    
    
    data = Data(x=merge_x, y=merge_y, edge_index=merge_edge_index)
    data.name = raw.name
    data.num_tasks = raw.num_tasks
    data.xe = merge_xe
    data.pretrain_edge_index = merge_pretrain_edge_index
    data.node_graph_id_map = torch.tensor(node_graph_id_map)
    data.edge_graph_id_map = torch.tensor(edge_graph_id_map)
    data.node_text_feat = raw.data.node_embs
    data.edge_text_feat = raw.data.edge_embs
    
    
    return data




# graph set partition
def graph_set_partition(data_tag, num_partitions, root=None, mode="pretrain"):
    dataset_dir = os.path.join(root, f"{data_tag.name.lower()}", f"tot_clients_{num_partitions}")
    os.makedirs(dataset_dir, exist_ok=True)
    pt_filename_list = [os.path.join(dataset_dir, f"client_{i}_pt.pt") for i in range(num_partitions)]
    ft_filename_list = [os.path.join(dataset_dir, f"client_{i}_ft.pt") for i in range(num_partitions)]
    
    all_partition_found = True
    for filename in pt_filename_list:
        if not os.path.exists(filename):
            all_partition_found = False
            break
        
    for filename in ft_filename_list:
        if not os.path.exists(filename) and data_tag.name.lower() not in ["chemblpre"]:
            all_partition_found = False
            break
        
        
    local_data = []
    
       
    if not all_partition_found:
        num_tot_graphs = len(data_tag) # 41127
        num_graphs_per_partition = int(num_tot_graphs / num_partitions)
        
        if data_tag.name.lower() in ["chemhiv","chempcba"]:
            split = "0.8-0.1-0.1"
        elif data_tag.name.lower() in ["chemblpre"]:
            split = "1.0-0-0"
        elif data_tag.name.lower() in ["proteins", "imdb-binary"]:
            split = "0.8-0.1-0.1"
        
        train_, val_, test_ = extract_floats(split)
        
        
        for partition_i, pt_filename in enumerate(pt_filename_list):
            print(f"processing {data_tag.name.lower()} client {partition_i}...")
            start_idx = partition_i * num_graphs_per_partition
            end_idx = (partition_i + 1) * num_graphs_per_partition if partition_i != num_partitions - 1 else num_tot_graphs
            
            
            data = data_tag[start_idx:end_idx]
            
            # split
            num_graphs = end_idx-start_idx
            graph_list = list(range(num_graphs))

            np.random.shuffle(graph_list)
            
            train_mask = idx_to_mask_tensor(graph_list[:int(train_ * num_graphs)], num_graphs).bool()
            val_mask = idx_to_mask_tensor(graph_list[int(train_ * num_graphs) : int((train_+val_) * num_graphs)], num_graphs).bool()
            test_mask = idx_to_mask_tensor(graph_list[int((train_+val_) * num_graphs): min(num_graphs, int((train_+val_+test_) * num_graphs))], num_graphs).bool()
            
            
            
            # save finetune file
            if data_tag.name.lower() not in ["chemblpre"]:
                data.name = data_tag.name.lower()
                data.weight = data_tag.weight
                data.node_embs = data_tag.node_embs
                data.node_text_feat = data_tag.node_embs
                data.edge_text_feat = data_tag.edge_embs
                data.class_node_text_feat = data_tag.class_node_text_feat
                data.train_mask = train_mask
                data.val_mask = val_mask
                data.test_mask = test_mask                
                data.node_text = data.texts[0]
                data.edge_text = data.texts[1]
                
                
                
                torch.save(data, ft_filename_list[partition_i])
        

            
            # save pretrain file
            pt_data = merge_raw(data)
            pt_data.train_mask = train_mask
            pt_data.val_mask = val_mask
            pt_data.test_mask = test_mask
            pt_data.node_text = data.texts[0]
            pt_data.edge_text = data.texts[1]
            torch.save(pt_data, pt_filename)

        
    # use cache
    if mode == "pretrain":
        for filename in pt_filename_list:
            local_data.append(torch.load(filename))
    else:
        for filename in ft_filename_list:
            local_data.append(torch.load(filename))
            
            
    return local_data


