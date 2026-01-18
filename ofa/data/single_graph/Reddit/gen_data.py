import os
import torch
import torch_geometric as pyg
import numpy as np
import networkx as nx


def get_data(dset):
    """
    加载Reddit数据集
    Reddit是社交网络数据集，包含41个子论坛（类别）
    节点代表Reddit帖子，边代表帖子之间的回复关系
    """
    cur_path = os.path.dirname(__file__)
    path = os.path.join(cur_path, "reddit.pt")
    
    # 如果.pt文件不存在，尝试从PyG下载
    if not os.path.exists(path):
        print(f"reddit.pt文件不存在，尝试生成...")
        
        # 从PyG下载Reddit数据集
        try:
            from torch_geometric.datasets import Reddit
            
            temp_dir = os.path.join(cur_path, 'temp_data')
            os.makedirs(temp_dir, exist_ok=True)
            print("正在从PyTorch Geometric加载Reddit数据集...")
            
            # PyG会在root目录下创建Reddit子目录
            actual_download_dir = os.path.join(temp_dir, 'Reddit')
            print(f"下载内容存放目录: {os.path.abspath(actual_download_dir)}")
            
            # 使用PyG的Reddit数据集
            dataset = Reddit(root=temp_dir)
            data = dataset[0]
            
            # 输出实际下载目录（PyG会在root下创建Reddit子目录）
            actual_download_path = os.path.join(temp_dir, 'Reddit')
            if os.path.exists(actual_download_path):
                print(f"数据集下载目录: {os.path.abspath(actual_download_path)}")
                # 列出下载目录中的文件
                if os.path.isdir(actual_download_path):
                    files = os.listdir(actual_download_path)
                    if files:
                        print(f"  下载的文件: {', '.join(files[:5])}{'...' if len(files) > 5 else ''}")
            
            print(f"数据集信息：")
            print(f"  节点数: {data.num_nodes}")
            print(f"  边数: {data.num_edges}")
            print(f"  特征维度: {data.num_features}")
            print(f"  类别数: {dataset.num_classes}")
            
            # 获取类别名称
            num_classes = dataset.num_classes
            label_names = [f'Subreddit_{i}' for i in range(num_classes)]
            
            # 生成节点文本
            raw_texts = [f"Reddit post {i} in subreddit {label_names[data.y[i].item()]}" 
                         for i in range(data.num_nodes)]
            
            # 添加必要的属性
            data.raw_texts = raw_texts
            data.label_names = label_names
            
            # 处理数据分割掩码（参考Photo/Computers的方式）
            # Reddit数据集通常已经提供了train_mask, val_mask, test_mask
            # 但为了与其他数据集保持一致，我们生成20个随机split
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
            
            # 保持mask为list格式，与其他数据集一致
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
            print(f"Reddit数据已处理并保存到: {path}")
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
            raise FileNotFoundError(f"无法加载Reddit数据集: {e}")
    else:
        print(f"使用已有的reddit.pt文件")
    
    # PyTorch 2.6+需要显式设置weights_only=False
    data = torch.load(path, weights_only=False)
    text = data.raw_texts if hasattr(data, 'raw_texts') else [f'Reddit post {i}' for i in range(data.x.shape[0])]
    label_names = data.label_names if hasattr(data, 'label_names') else [f'Subreddit_{i}' for i in range(41)]
    
    # 如果masks是stacked tensor，转换为list格式（兼容旧数据）
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
    print(f"Reddit edge_index size: {edge_index.size()}")
    
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    new_data = pyg.data.data.Data(**data_dict)
    
    # 显式复制train_masks, val_masks, test_masks等属性到新对象
    if hasattr(data, 'train_masks'):
        new_data.train_masks = data.train_masks
    if hasattr(data, 'val_masks'):
        new_data.val_masks = data.val_masks
    if hasattr(data, 'test_masks'):
        new_data.test_masks = data.test_masks
    
    # 构造文本描述
    clean_text = ["feature node. reddit post description: " + str(t) for t in text]
    label_text = [
        "prompt node. subreddit category: " + label
        for label in label_names
    ]
    edge_label_text = [
        "prompt node. two posts are not connected by replies",
        "prompt node. two posts are connected by replies"
    ]
    edge_text = [
        "feature edge. connected posts are linked by reply relationships."
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on posts that are connected by replies"
    ]
    noi_node_text = [
        "prompt node. node classification on the post's subreddit category"
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

