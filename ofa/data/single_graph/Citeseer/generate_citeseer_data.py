"""
【新增脚本】生成Citeseer数据集的.pt文件
此脚本将从PyTorch Geometric加载Citeseer数据集，并保存为与Cora相同格式的.pt文件

运行方法：
    python generate_citeseer_data.py

如果遇到网络问题，可以：
1. 使用代理：export http_proxy=... && export https_proxy=...
2. 手动下载Citeseer数据集到指定目录
3. 使用--offline参数和已有的数据路径
"""

import os
import sys
import torch
import argparse

def generate_citeseer_data(offline_data_path=None):
    """
    从PyG加载Citeseer数据集并保存为.pt格式
    
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
        print("正在从PyTorch Geometric加载Citeseer数据集...")
    
    try:
        import torch_geometric as pyg
        from torch_geometric.datasets import Planetoid
        
        # 使用PyG的Planetoid数据集加载Citeseer
        dataset = Planetoid(root=temp_dir, name='Citeseer')
        data = dataset[0]
        
        print(f"数据集信息：")
        print(f"  节点数: {data.num_nodes}")
        print(f"  边数: {data.num_edges}")
        print(f"  特征维度: {data.num_features}")
        print(f"  类别数: {dataset.num_classes}")
        
    except Exception as e:
        print(f"\n错误: 无法加载Citeseer数据集")
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
    
    # 获取类别名称（Citeseer的6个类别）
    label_names = ['Agents', 'AI', 'DB', 'IR', 'ML', 'HCI']
    
    # 生成节点文本（论文标题和摘要的占位符）
    # 注意：Citeseer原始数据没有文本，这里生成占位符
    raw_texts = [f"Paper {i} in category {label_names[data.y[i].item()]}" 
                 for i in range(data.num_nodes)]
    
    # 添加必要的属性
    data.raw_texts = raw_texts
    data.label_names = label_names
    
    # 处理数据分割掩码
    # PyG的Citeseer数据集提供了train_mask, val_mask, test_mask
    # 我们需要将它们转换为与Cora相同的格式（多个split）
    
    # 创建多个随机split（类似Cora的处理方式）
    num_splits = 20  # 与Cora保持一致
    train_masks = []
    val_masks = []
    test_masks = []
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    
    for i in range(num_splits):
        if i == 0:
            # 第一个split使用原始的split
            train_masks.append(data.train_mask.clone())
            val_masks.append(data.val_mask.clone())
            test_masks.append(data.test_mask.clone())
        else:
            # 其他split使用随机划分
            perm = torch.randperm(data.num_nodes)
            num_train = data.train_mask.sum().item()
            num_val = data.val_mask.sum().item()
            
            train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
            
            train_mask[perm[:num_train]] = True
            val_mask[perm[num_train:num_train+num_val]] = True
            test_mask[perm[num_train+num_val:]] = True
            
            train_masks.append(train_mask)
            val_masks.append(val_mask)
            test_masks.append(test_mask)
    
    # 【关键修复】保持mask为list格式，与Cora数据格式一致
    # 不要使用torch.stack，直接保存为list of tensors
    data.train_masks = train_masks  # list of tensors
    data.val_masks = val_masks      # list of tensors
    data.test_masks = test_masks    # list of tensors
    
    # 删除单一的mask（保留多split版本）
    if hasattr(data, 'train_mask'):
        delattr(data, 'train_mask')
    if hasattr(data, 'val_mask'):
        delattr(data, 'val_mask')
    if hasattr(data, 'test_mask'):
        delattr(data, 'test_mask')
    
    # 保存数据
    output_path = os.path.join(current_dir, 'citeseer.pt')
    torch.save(data, output_path)
    print(f"\nCiteseer数据已成功保存到: {output_path}")
    
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
    parser = argparse.ArgumentParser(description='生成Citeseer数据集')
    parser.add_argument('--offline', type=str, default=None,
                      help='离线数据路径（如果已经下载）')
    args = parser.parse_args()
    
    success = generate_citeseer_data(args.offline)
    sys.exit(0 if success else 1)
