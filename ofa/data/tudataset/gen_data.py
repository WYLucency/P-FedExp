import os
import torch
import torch_geometric as pyg
import numpy as np
from torch_geometric.datasets import TUDataset
from ofa.data.ofa_data import OFAPygDataset


class TUDatasetOFADataset(OFAPygDataset):
    """
    OFA Dataset wrapper for TUDataset (PROTEINS, IMDB-BINARY, etc.)
    These datasets don't have native text descriptions, so we mock text features.
    """
    
    def __init__(self, name: str, encoder, root: str = "./cache_data", load_text: bool = True,
                 transform=None, pre_transform=None):
        # TUDataset names in PyG are uppercase (e.g., "PROTEINS", "IMDB-BINARY")
        self.tu_name = name.upper()
        super().__init__(name, encoder, root, load_text, transform, pre_transform)
    
    def gen_data(self):
        """
        Generate data from TUDataset.
        Since TUDataset has no native text, we create mock text features.
        """
        print(f"Loading TUDataset: {self.tu_name}")
        
        # Load TUDataset from PyTorch Geometric
        # TUDataset will download automatically if not present
        tu_dataset = TUDataset(root=os.path.join(self.root, "raw"), name=self.tu_name)
        
        data_list = []
        node_texts = ["node"]  # Mock node text
        edge_texts = ["edge"]  # Mock edge text
        
        # Get number of classes (collect all labels first)
        all_labels = []
        for data in tu_dataset:
            if data.y.dim() == 0:
                all_labels.append(data.y.item())
            else:
                all_labels.extend(data.y.tolist())
        num_classes = len(torch.unique(torch.tensor(all_labels)))
        labels_features = [f"class_{i}" for i in range(num_classes)]
        
        # Convert TUDataset graphs to our format
        # Similar to MolOFADataset:
        # - data.x: indices to unique node texts (all nodes use index 0 since we only have "node")
        # - pretrain_edge_index: global accumulated edge index for pretraining
        num_nodes = 0
        for i, data in enumerate(tu_dataset):
            # Create node and edge indices
            num_graph_nodes = data.num_nodes
            
            # All nodes use the same text "node", so all use index 0
            # This is similar to MolOFADataset where x indexes into unique node texts
            node_ids = torch.zeros(num_graph_nodes, dtype=torch.long)
            
            # Edge indices (already in correct format, local to this graph)
            edge_index = data.edge_index
            
            # Create pretrain_edge_index (shifted by cumulative node count for global indexing)
            pretrain_edge_index = edge_index + num_nodes
            
            # Create edge type indices (all edges use type 0, indexing into edge_texts)
            edge_type_ids = torch.zeros(edge_index.shape[1], dtype=torch.long)
            
            # Create data object
            pyg_data = pyg.data.Data(
                x=node_ids,  # Indices to unique node texts (all 0)
                xe=edge_type_ids,  # Indices to unique edge texts (all 0)
                edge_index=edge_index,  # Local edge index
                pretrain_edge_index=pretrain_edge_index,  # Global edge index
                y=data.y.view(-1)  # Ensure y is 1D
            )
            
            data_list.append(pyg_data)
            num_nodes += num_graph_nodes
        
        # Create split (will be handled by MolSplitter or manually in basic_utils.py)
        split = {"train": [], "valid": [], "test": []}
        
        # Prompt texts (similar to MolOFADataset)
        prompt_edge_text = [
            "prompt edge.",
            "prompt edge. edge for query graph that is our target",
            "prompt edge. edge for support graph that is an example",
        ]
        prompt_text = [
            "prompt node. graph classification",
            "prompt node. few shot task node for graph classification that decides whether the query graph belongs to the class of support graphs.",
        ]
        
        prompt_text_map = {
            "e2e_graph": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(labels_features))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
            },
            "lr_graph": {
                "noi_node_text_feat": ["noi_node_text_feat", [1]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(labels_features))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]
            }
        }
        
        texts = [node_texts, edge_texts, labels_features, prompt_edge_text, prompt_text]
        side_data = [split, prompt_text_map]
        
        return data_list, texts, side_data
    
    def add_text_emb(self, data_list, text_emb):
        """
        Add text embeddings to the data.
        Since we're mocking text features, we'll create embeddings from the mock texts.
        """
        data, slices = self.collate(data_list)
        
        # Store embeddings
        data.node_embs = text_emb[0]  # Node embeddings
        data.edge_embs = text_emb[1]  # Edge embeddings
        data.class_node_text_feat = text_emb[2]  # Class node text features
        
        return data, slices
    
    def get(self, index):
        """Get a single graph with text features."""
        data = super().get(index)
        
        # Map node indices to embeddings
        node_feat = self.data.node_embs[data.x]
        edge_feat = self.data.edge_embs[data.xe]
        
        data.node_text_feat = node_feat
        data.edge_text_feat = edge_feat
        
        # Ensure y is in the right format
        if data.y.dim() > 1:
            data.y = data.y.view(-1)
        
        return data
    
    def get_idx_split(self):
        """Get train/valid/test split indices."""
        return self.side_data[0]
    
    def get_task_map(self):
        """Get task mapping."""
        return self.side_data[1]
    
    def get_edge_list(self, mode="e2e"):
        """Get edge list for different modes."""
        if mode == "e2e_graph":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0]}
        elif mode == "lr_graph":
            return {"f2n": [1, 0], "n2f": [3, 0]}

