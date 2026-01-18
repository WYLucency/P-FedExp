"""
【新增脚本】生成Computers数据集的.pt文件
此脚本将从PyTorch Geometric加载Amazon Computers数据集，并保存为与Citeseer相同格式的.pt文件

运行方法：
    python generate_computers_data.py

如果遇到网络问题，可以：
1. 使用代理：export http_proxy=... && export https_proxy=...
2. 手动下载Computers数据集到指定目录
3. 使用--offline参数和已有的数据路径
"""

import os
import sys
import torch
import argparse

def generate_computers_data(offline_data_path=None):
    """
    从PyG加载Amazon Computers数据集并保存为.pt格式
    
    Args:
        offline_data_path: 如果提供，将从该路径加载已下载的数据
    """
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 创建临时目录用于下载原始数据
    if offline_data_path and os.path.exists(offline_data_path):
        temp_dir = offline_data_path
        print(f"使用离线数据: {temp_dir}")
    else:
        temp_dir = os.path.join(current_dir, 'temp_data')
        os.makedirs(temp_dir, exist_ok=True)
        print("正在从PyTorch Geometric加载Amazon Computers数据集...")
    
    try:
        import torch_geometric as pyg
        from torch_geometric.datasets import Amazon
        
        # 使用PyG的Amazon数据集加载Computers（参考Citeseer加前缀的方式）
        dataset = Amazon(root=temp_dir, name='Computers')
        data = dataset[0]
        
        print(f"数据集信息：")
        print(f"  节点数: {data.num_nodes}")
        print(f"  边数: {data.num_edges}")
        print(f"  特征维度: {data.num_features}")
        print(f"  类别数: {dataset.num_classes}")
        
    except Exception as e:
        print(f"\n错误: 无法加载Computers数据集")
        print(f"错误信息: {e}")
        print("\n解决方案:")
        print("1. 检查网络连接")
        print("2. 使用代理: export http_proxy=http://your_proxy:port")
        print("3. 手动下载数据集到临时目录")
        print(f"   目标目录: {temp_dir}")
        print("\n或者使用以下命令在conda环境中安装:")
        print("   conda activate tor2.1+cu12.1")
        print("   pip install torch-geometric")
        return False
    
    # 获取类别名称（Computers有10个类别）
    num_classes = dataset.num_classes
    label_names = [f'Category_{i}' for i in range(num_classes)]
    
    # 生成节点文本（产品描述的占位符）
    raw_texts = [f"Product {i} in category {label_names[data.y[i].item()]}" 
                 for i in range(data.num_nodes)]
    
    # 添加必要的属性
    data.raw_texts = raw_texts
    data.label_names = label_names
    
    # 处理数据分割掩码
    # Amazon数据集通常不提供预定义的train/val/test mask
    # 我们需要创建多个随机split（类似Citeseer的处理方式）
    
    # 创建多个随机split（类似Citeseer的处理方式）
    num_splits = 20  # 与Citeseer保持一致
    train_masks = []
    val_masks = []
    test_masks = []
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    # 计算每个类别的节点数，用于分层采样
    num_nodes = data.num_nodes
    num_train_per_class = max(1, int(0.2 * num_nodes / num_classes))
    num_val_per_class = max(1, int(0.4 * num_nodes / num_classes))
    
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
                
                # 分割: 20% train, 40% val, 40% test (参考Citeseer的分割)
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
    
    # 保存数据
    output_path = os.path.join(current_dir, 'computers.pt')
    torch.save(data, output_path)
    print(f"\nComputers数据已成功保存到: {output_path}")
    
    # 清理临时文件（仅在非离线模式下）
    if not offline_data_path:
        import shutil
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print("临时文件已清理")
            except:
                print(f"警告: 无法删除临时目录 {temp_dir}，请手动清理")
    
    print("\n✓ 数据生成完成!")
    print("下一步：运行pretrain_fedbook.py进行预训练")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成Computers数据集')
    parser.add_argument('--offline', type=str, default=None,
                      help='离线数据路径（如果已经下载）')
    args = parser.parse_args()
    
    success = generate_computers_data(args.offline)
    sys.exit(0 if success else 1)

