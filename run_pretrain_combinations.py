"""
遍历预训练的不同数据集组合，并添加_fedbook后缀表示fedbook Baseline

使用方法:
    python run_pretrain_combinations.py [--gpu_id 0] [--seed 2025] [--auto_combinations]

参数:
    --gpu_id: GPU ID (默认: 0)
    --seed: 随机种子 (默认: 2025)
    --auto_combinations: 自动生成所有可能的组合（默认: False，使用预定义的组合）
    --min_size: 最小组合大小（仅在--auto_combinations时有效，默认: 1）
    --max_size: 最大组合大小（仅在--auto_combinations时有效，默认: 3）
"""
import os
import json
import subprocess
import itertools
import os.path as osp
import argparse
import sys

# 可用的数据集列表
# AVAILABLE_DATASETS = ["cora", "citeSeer", "pubmed", "photo", "computers", "wikics"]
AVAILABLE_DATASETS = ["chempcba", "chemhiv","proteins","imdb-binary"]

# ["citeseer", "pubmed", "computers", "wikics", "photo"],
    # ["cora", "pubmed", "computers", "wikics", "photo"],
    # ["cora", "citeseer", "computers", "wikics", "photo"],
    # ["cora", "citeseer", "pubmed", "computers", "wikics"],
PRETRAIN_COMBINATIONS = [
    ["chempcba", "chemhiv","imdb-binary"],
    ["chempcba", "chemhiv","proteins"]
]

# 每个数据集的客户端数量（可以根据需要调整）
NUM_CLIENTS_PER_DATASET = 1

# 基础配置路径
BASE_CONFIG_PATH = osp.join(osp.dirname(__file__), "config", "pretrain_config.json")
SCRIPT_DIR = osp.dirname(__file__)

def create_config_for_combination(datasets, output_path):
    """为给定的数据集组合创建配置文件
    
    Args:
        datasets: 预训练使用的数据集列表
        output_path: 输出配置文件路径
    """
    # 找出没有参与预训练的数据集（用于微调）
    datasets_lower = [ds.lower() for ds in datasets]
    finetune_datasets = [ds for ds in AVAILABLE_DATASETS if ds.lower() not in datasets_lower]
    
    if len(finetune_datasets) == 0:
        # 如果所有数据集都参与了预训练，则使用第一个数据集作为微调（fallback）
        finetune_datasets = [AVAILABLE_DATASETS[0]]
        print(f"警告: 所有数据集都参与了预训练，使用 {finetune_datasets[0]} 作为微调数据集")
    else:
        print(f"预训练数据集: {datasets}, 微调数据集: {finetune_datasets}")
    
    config = {
        "root": osp.dirname(__file__),
        "_comment_pretrain": f"【预训练配置】{len(datasets)}个数据集组合: {', '.join(datasets)}",
        "pretrain_num_clients": [NUM_CLIENTS_PER_DATASET] * len(datasets),
        "_comment_finetune": f"【微调配置】在{', '.join(finetune_datasets)}数据集上进行微调（未参与预训练的数据集）",
        "finetune_num_clients": [1] * len(finetune_datasets),
        "_comment_tasks": "【任务说明】预训练和微调都是节点分类任务",
        "tasks": ["graph_cls"] * len(datasets),
        "_comment_datasets": f"【预训练数据集】{', '.join(datasets)}",
        "datasets": datasets,
        "_comment_weights": "【权重说明】预训练时各数据集权重相等",
        "weights": [1] * len(datasets),
        "_comment_finetune_datasets": f"【微调数据集】{', '.join(finetune_datasets)}（未参与预训练的数据集）",
        "finetune_datasets": finetune_datasets,
        "finetune_tasks": ["graph_cls"] * len(finetune_datasets)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    return config

def get_model_path_suffix(datasets, num_clients_list):
    """生成模型路径后缀（带_fedbook）"""
    suffix = "_".join([f"{ds.lower()}_{nc}" for ds, nc in zip(datasets, num_clients_list)])
    return f"{suffix}_fedbook"

def run_pretrain(config_path, gpu_id=0, seed=2025):
    """运行预训练脚本"""
    cmd = [
        sys.executable, 
        osp.join(SCRIPT_DIR, "pretrain_fedbook.py"),
        "--data_config", config_path,
        "--gpu_id", str(gpu_id),
        "--seed", str(seed)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running pretrain with config: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return result.returncode == 0

def generate_all_combinations(min_size=1, max_size=3):
    """自动生成所有可能的数据集组合"""
    combinations = []
    for size in range(min_size, max_size + 1):
        for combo in itertools.combinations(AVAILABLE_DATASETS, size):
            combinations.append(list(combo))
    return combinations

def main():
    """主函数：遍历所有组合并运行预训练"""
    parser = argparse.ArgumentParser(description='遍历预训练组合（fedbook Baseline）')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--auto_combinations", action="store_true", help="自动生成所有可能的组合")
    parser.add_argument("--min_size", type=int, default=1, help="最小组合大小（仅在--auto_combinations时有效）")
    parser.add_argument("--max_size", type=int, default=3, help="最大组合大小（仅在--auto_combinations时有效）")
    
    args = parser.parse_args()
    
    # 确定要使用的组合
    if args.auto_combinations:
        combinations = generate_all_combinations(args.min_size, args.max_size)
        print(f"自动生成组合模式: 最小大小={args.min_size}, 最大大小={args.max_size}")
    else:
        combinations = PRETRAIN_COMBINATIONS
        print("使用预定义的组合")
    
    print("="*60)
    print("开始遍历预训练组合（fedbook Baseline）")
    print(f"总共有 {len(combinations)} 个组合需要运行")
    print("="*60)
    
    # 创建临时配置目录
    temp_config_dir = osp.join(SCRIPT_DIR, "config", "temp_pretrain_configs")
    os.makedirs(temp_config_dir, exist_ok=True)
    
    results = []
    
    for idx, datasets in enumerate(combinations):
        print(f"\n{'#'*60}")
        print(f"组合 {idx+1}/{len(combinations)}: {datasets}")
        print(f"{'#'*60}")
        
        # 创建临时配置文件
        config_name = "_".join(datasets) + ".json"
        temp_config_path = osp.join(temp_config_dir, config_name)
        create_config_for_combination(datasets, temp_config_path)
        
        # 运行预训练
        success = run_pretrain(temp_config_path, gpu_id=args.gpu_id, seed=args.seed)
        
        results.append({
            "datasets": datasets,
            "config_path": temp_config_path,
            "success": success
        })
        
        if success:
            print(f"✓ 组合 {datasets} 预训练完成")
        else:
            print(f"✗ 组合 {datasets} 预训练失败")
    
    # 打印总结
    print("\n" + "="*60)
    print("预训练组合运行总结")
    print("="*60)
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    print(f"总计: {len(results)} 个组合")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print("\n详细结果:")
    for idx, result in enumerate(results):
        status = "✓ 成功" if result["success"] else "✗ 失败"
        print(f"{idx+1}. {result['datasets']}: {status}")
    
    print(f"\n所有配置文件保存在: {temp_config_dir}")
    print("="*60)

if __name__ == "__main__":
    main()

