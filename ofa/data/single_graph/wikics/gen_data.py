import functools
import json
import os

import torch
from torch_geometric.data import Data


def get_text(path):
    """
    Returns: node_text_lst, label_text_lst
    Node text format: "wikipedia entry name: xxx. entry content: xxxxx"
    Label text format: "wikipedia entry category: xxx"
    """
    with open(os.path.join(path, "metadata.json")) as json_file:
        raw_data = json.load(json_file)

    node_info = raw_data["nodes"]
    label_info = raw_data["labels"]
    node_text_lst = []
    label_text_lst = []

    # Process Node Feature
    for node in node_info:
        node_feature = ((
                "feature node. wikipedia entry name: " + node["title"] + ". entry content: " + functools.reduce(
            lambda x, y: x + " " + y, node["tokens"])).lower().strip())
        node_text_lst.append(node_feature)

    # Process Label Feature
    for label in label_info.values():
        label_feature = (("prompt node. wikipedia entry category: " + label).lower().strip())
        label_text_lst.append(label_feature)

    return node_text_lst, label_text_lst


def load_wikics_from_local(path):
    """
    Load WikiCS data from local data.json file to avoid downloading
    """
    data_path = os.path.join(path, "data.json")
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    
    # Convert adjacency list to edge_index
    # data_dict['links'] is in adjacency list format: each row contains neighbors of that node
    edge_list = []
    for node_idx, neighbors in enumerate(data_dict['links']):
        for neighbor in neighbors:
            edge_list.append([node_idx, neighbor])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    x = torch.tensor(data_dict['features'], dtype=torch.float)
    y = torch.tensor(data_dict['labels'], dtype=torch.long)
    
    # Load train/val/test splits
    train_mask = torch.tensor(data_dict['train_masks'], dtype=torch.bool).t()
    val_mask = torch.tensor(data_dict['val_masks'], dtype=torch.bool).t()
    stopping_mask = torch.tensor(data_dict['stopping_masks'], dtype=torch.bool).t()
    test_mask = torch.tensor(data_dict['test_mask'], dtype=torch.bool)
    
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=train_mask, val_mask=val_mask, 
                stopping_mask=stopping_mask, test_mask=test_mask)
    
    return data


def get_data(dset):
    # Load data from local data.json instead of downloading
    cur_path = os.path.dirname(__file__)
    pyg_data_obj = load_wikics_from_local(cur_path)
    
    node_texts, label_texts = get_text(cur_path)
    edge_text = ["feature edge. wikipedia page link"]
    prompt_text = ["prompt node. node classification of wikipedia entry category"]
    prompt_edge_text = ["prompt edge."]
    prompt_text_map = {"e2e_node": {"noi_node_text_feat": ["noi_node_text_feat", [0]],
                                    "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_texts))],
                                    "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]}}
    return ([pyg_data_obj], [node_texts, edge_text, prompt_text, label_texts, prompt_edge_text, ], prompt_text_map,)
