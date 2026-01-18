import os
import pandas as pd
import torch
import torch_geometric as pyg
import numpy as np
from scipy.sparse import csr_matrix


def load_npz_to_sparse_tensor(npz_file):
    """从.npz文件直接加载稀疏矩阵数据"""
    with np.load(npz_file, allow_pickle=True) as loader:
        # 加载邻接矩阵
        adj_matrix = csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                shape=loader['adj_shape'])
        # 加载特征矩阵
        attr_matrix = csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                 shape=loader['attr_shape'])
        # 加载标签
        labels = loader['labels']
        # 加载类别名称
        class_names = loader.get('class_names', None)
        
    return adj_matrix, attr_matrix, labels, class_names


def get_data(dset):
    """
    加载Amazon Photo数据集
    Photo是Amazon产品共购买网络的一部分，包含8个产品类别
    """
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, "photo.pt")
    
    # 如果.pt文件不存在，尝试从PyG下载或从.npz文件加载
    if not os.path.exists(path):
        print(f"photo.pt文件不存在，尝试生成...")
        
        # 首先尝试从PyG下载（参考Citeseer的方式）
        try:
            from torch_geometric.datasets import Amazon
            
            temp_dir = os.path.join(cur_path, 'temp_data')
            os.makedirs(temp_dir, exist_ok=True)
            print("正在从PyTorch Geometric加载Amazon Photo数据集...")
            
            # 使用PyG的Amazon数据集加载Photo（参考Citeseer加前缀的方式）
            dataset = Amazon(root=temp_dir, name='Photo')
            data = dataset[0]
            
            print(f"数据集信息：")
            print(f"  节点数: {data.num_nodes}")
            print(f"  边数: {data.num_edges}")
            print(f"  特征维度: {data.num_features}")
            print(f"  类别数: {dataset.num_classes}")
            
            # 获取类别名称
            num_classes = dataset.num_classes
            label_names = [f'Category_{i}' for i in range(num_classes)]
            
            # 生成节点文本
            raw_texts = [f"Product {i} in category {label_names[data.y[i].item()]}" 
                         for i in range(data.num_nodes)]
            
            # 添加必要的属性
            data.raw_texts = raw_texts
            data.label_names = label_names
            
            # 处理数据分割掩码（参考Citeseer的方式）
            num_splits = 20
            train_masks = []
            val_masks = []
            test_masks = []
            
            # 设置随机种子以确保可重复性
            torch.manual_seed(42)
            
            num_nodes = data.num_nodes
            for i in range(num_splits):
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                # 为每个类别随机分配节点到train/val/test
                for class_id in range(num_classes):
                    class_indices = (data.y == class_id).nonzero(as_tuple=True)[0]
                    num_class_nodes = len(class_indices)
                    
                    if num_class_nodes > 0:
                        # 随机打乱
                        perm = torch.randperm(num_class_nodes)
                        shuffled_indices = class_indices[perm]
                        
                        # 分割: 20% train, 40% val, 40% test
                        train_size = max(1, int(0.2 * num_class_nodes))
                        val_size = max(1, int(0.4 * num_class_nodes))
                        
                        train_mask[shuffled_indices[:train_size]] = True
                        val_mask[shuffled_indices[train_size:train_size + val_size]] = True
                        test_mask[shuffled_indices[train_size + val_size:]] = True
                
                train_masks.append(train_mask)
                val_masks.append(val_mask)
                test_masks.append(test_mask)
            
            # 【关键修复】保持mask为list格式，与Citeseer数据格式一致
            # 不要使用torch.stack，直接保存为list of tensors
            data.train_masks = train_masks  # list of tensors
            data.val_masks = val_masks      # list of tensors
            data.test_masks = test_masks    # list of tensors
            
            # 删除单一的mask（如果存在）
            if hasattr(data, 'train_mask'):
                delattr(data, 'train_mask')
            if hasattr(data, 'val_mask'):
                delattr(data, 'val_mask')
            if hasattr(data, 'test_mask'):
                delattr(data, 'test_mask')
            
            # 保存为.pt文件
            torch.save(data, path)
            print(f"Photo数据已处理并保存到: {path}")
            print(f"节点数: {data.x.shape[0]}, 边数: {data.edge_index.shape[1]}, 特征维度: {data.x.shape[1]}")
            
            # 清理临时文件
            import shutil
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print("临时文件已清理")
                except:
                    print(f"警告: 无法删除临时目录 {temp_dir}，请手动清理")
        
        except Exception as e:
            print(f"从PyG下载失败: {e}")
            print("尝试从.npz文件加载...")
            
            # 如果PyG下载失败，尝试从.npz文件加载
            pyg_raw_dir = os.path.join(cur_path, "raw")
            npz_variants = [
                os.path.join(pyg_raw_dir, "amazon_electronics_photo.npz"),
                os.path.join(cur_path, "amazon_electronics_photo.npz"),
            ]
            
            npz_file = None
            for variant in npz_variants:
                if os.path.exists(variant) and os.path.getsize(variant) > 1000:
                    npz_file = variant
                    print(f"找到数据文件: {npz_file}")
                    break
            
            if npz_file is None:
                raise FileNotFoundError("未找到Photo数据集的.npz文件，也无法从PyG下载")
            
            # 直接从.npz加载数据
            adj_matrix, attr_matrix, labels, class_names = load_npz_to_sparse_tensor(npz_file)
            
            # 转换为PyG格式
            edge_index = torch.from_numpy(np.vstack(adj_matrix.nonzero())).long()
            x = torch.from_numpy(attr_matrix.todense()).float()
            y = torch.from_numpy(labels).long()
            
            # 创建PyG Data对象
            data = pyg.data.Data(x=x, edge_index=edge_index, y=y)
            
            # 添加必要的属性
            if class_names is not None:
                data.label_names = [str(name) for name in class_names]
            else:
                data.label_names = [f'Category_{i}' for i in range(8)]
            
            data.raw_texts = [f'Product {i}' for i in range(data.x.shape[0])]
            
            # 生成train/val/test masks (参考Citeseer的处理方式，生成20个随机split)
            num_nodes = data.x.shape[0]
            num_classes = len(data.label_names)
            num_splits = 20
            
            train_masks = []
            val_masks = []
            test_masks = []
            
            # 设置随机种子以确保可重复性
            torch.manual_seed(42)
            
            for split_id in range(num_splits):
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                # 为每个类别随机分配节点到train/val/test
                for class_id in range(num_classes):
                    class_indices = (y == class_id).nonzero(as_tuple=True)[0]
                    num_class_nodes = len(class_indices)
                    
                    if num_class_nodes > 0:
                        # 随机打乱
                        perm = torch.randperm(num_class_nodes)
                        shuffled_indices = class_indices[perm]
                        
                        # 分割: 20% train, 40% val, 40% test
                        train_size = max(1, int(0.2 * num_class_nodes))
                        val_size = max(1, int(0.4 * num_class_nodes))
                        
                        train_mask[shuffled_indices[:train_size]] = True
                        val_mask[shuffled_indices[train_size:train_size + val_size]] = True
                        test_mask[shuffled_indices[train_size + val_size:]] = True
                
                train_masks.append(train_mask)
                val_masks.append(val_mask)
                test_masks.append(test_mask)
            
            # 【关键修复】保持mask为list格式，与Citeseer数据格式一致
            # 不要使用torch.stack，直接保存为list of tensors
            data.train_masks = train_masks  # list of tensors
            data.val_masks = val_masks      # list of tensors
            data.test_masks = test_masks    # list of tensors
            
            # 保存为.pt文件
            torch.save(data, path)
            print(f"Photo数据已处理并保存到: {path}")
            print(f"节点数: {data.x.shape[0]}, 边数: {data.edge_index.shape[1]}, 特征维度: {data.x.shape[1]}")
    else:
        print(f"使用已有的photo.pt文件")
    
    # PyTorch 2.6+需要显式设置weights_only=False
    data = torch.load(path, weights_only=False)
    text = data.raw_texts if hasattr(data, 'raw_texts') else [f'Product {i}' for i in range(data.x.shape[0])]
    label_names = data.label_names if hasattr(data, 'label_names') else [f'Category_{i}' for i in range(8)]
    
    # 【关键修复】如果masks是stacked tensor，转换为list格式（兼容旧数据）
    if hasattr(data, 'train_masks') and isinstance(data.train_masks, torch.Tensor):
        print("检测到stacked格式的masks，转换为list格式...")
        num_splits = data.train_masks.shape[0]
        data.train_masks = [data.train_masks[i] for i in range(num_splits)]
        data.val_masks = [data.val_masks[i] for i in range(num_splits)]
        data.test_masks = [data.test_masks[i] for i in range(num_splits)]
        # 重新保存以更新格式
        torch.save(data, path)
        print("已更新为list格式")
    
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    print(f"Photo edge_index size: {edge_index.size()}")
    
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    new_data = pyg.data.data.Data(**data_dict)
    
    # 【关键修复】显式复制train_masks, val_masks, test_masks等属性到新对象
    # 因为to_dict()可能不包含这些自定义属性
    if hasattr(data, 'train_masks'):
        new_data.train_masks = data.train_masks
    if hasattr(data, 'val_masks'):
        new_data.val_masks = data.val_masks
    if hasattr(data, 'test_masks'):
        new_data.test_masks = data.test_masks
    
    # 构造文本描述
    clean_text = ["feature node. product description: " + str(t) for t in text]
    label_text = [
        "prompt node. product category: " + label
        for label in label_names
    ]
    edge_label_text = [
        "prompt node. two products are not frequently bought together",
        "prompt node. two products are frequently bought together"
    ]
    edge_text = [
        "feature edge. connected products are frequently bought together."
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on products that are bought together"
    ]
    noi_node_text = [
        "prompt node. node classification on the product's category"
    ]
    prompt_edge_text = [
        "prompt edge", 
        "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example"
    ]
    
    return (
        [new_data],
        [
            clean_text,
            edge_text,
            noi_node_text + noi_node_edge_text,
            label_text + edge_label_text,
            prompt_edge_text,
        ],
        {
            "e2e_node": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
            },
            "e2e_link": {
                "noi_node_text_feat": ["noi_node_text_feat", [1]],
                "class_node_text_feat": ["class_node_text_feat",
                                         torch.arange(len(label_text), len(label_text) + len(edge_label_text))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]]
            },
            "lr_node": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text))],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]]
            },
        }
    )

