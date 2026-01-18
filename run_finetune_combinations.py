"""
遍历微调的不同预训练配置和 kshot 值（fedbook Baseline）

使用方法:
    python run_finetune_combinations.py [--gpu_id 0] [--seed 2025] [--config_dir <dir>]

参数:
    --gpu_id: GPU ID (默认: 0)
    --seed: 随机种子 (默认: 2025)
    --config_dir: 预训练配置文件目录 (默认: config/temp_pretrain_configs)
    --kshot_list: kshot 值列表，用逗号分隔 (默认: -1,1,5)
    --standard: 每个实验运行的轮数 (默认: 10)
"""
import os
import json
import subprocess
import glob
import os.path as osp
import argparse
import time

# 默认配置
DEFAULT_CONFIG_DIR = "/home/zhuangzy25/FGLFGM/FedBook/archive/config/tempnew"
SCRIPT_DIR = "/home/zhuangzy25/FGLFGM/FedBook/archive"
LOG_BASE_DIR = "/home/zhuangzy25/FGLFGM/FedBook/archive/logs"
DEFAULT_KSHOT_LIST = [-1]
DEFAULT_STANDARD = 2

def find_pretrain_configs(config_dir):
    """查找所有预训练配置文件"""
    config_pattern = osp.join(config_dir, "*.json")
    config_files = glob.glob(config_pattern)
    return sorted(config_files)

def check_model_exists(config_path):
    """检查对应的预训练模型是否存在"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        datasets = config.get("datasets", [])
        num_clients = config.get("pretrain_num_clients", [])
        
        # 生成模型路径（fedbook版本）
        model_path_base = "_".join([f"{ds.lower()}_{nc}" for ds, nc in zip(datasets, num_clients)])
        model_path = osp.join(SCRIPT_DIR, 'ckpts', 'pretrain_models', f"{model_path_base}_fedbook")
        server_model_path = osp.join(model_path, "server", "encoder.pt")
        
        return osp.exists(server_model_path), model_path
    except Exception as e:
        print(f"检查模型时出错: {e}")
        return False, None

def run_finetune(config_path, kshot, standard, log_dir, gpu_id=0, seed=2025):
    """运行微调脚本"""
    cmd = [
        "python", 
        osp.join(SCRIPT_DIR, "finetune_fedbook.py"),
        "--data_config", config_path,
        "--k_shot", str(kshot),
        "--standard", str(standard),
        "--gpu_id", str(gpu_id),
        "--seed", str(seed),
        "--log_dir", log_dir
    ]
    
    print(f"\n{'='*60}")
    print(f"Running finetune:")
    print(f"  Config: {osp.basename(config_path)}")
    print(f"  K-shot: {kshot}")
    print(f"  Standard: {standard}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    try:
        # 运行微调脚本，实时显示输出
        result = subprocess.run(cmd, cwd=SCRIPT_DIR, timeout=None)
        
        # 方法1: 检查返回码
        if result.returncode == 0:
            return True
        
        # 方法2: 即使返回码非0，也检查日志文件是否包含成功标志
        # 查找对应的日志文件（可能包含时间戳，所以需要匹配模式）
        import glob
        log_pattern = osp.join(log_dir, f"FedBook_setting_kshot{kshot}_standard{standard}_*.log")
        log_files = glob.glob(log_pattern)
        
        if log_files:
            # 检查最新的日志文件
            latest_log = max(log_files, key=osp.getmtime)
            try:
                with open(latest_log, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    # 检查是否包含成功完成的标志
                    if "Finetune completed" in log_content and "FINAL RESULTS SUMMARY" in log_content:
                        print(f"\n注意: 脚本返回码为 {result.returncode}，但日志显示已完成")
                        print(f"检查日志文件: {latest_log}")
                        print("判断为成功完成\n")
                        return True
            except Exception as e:
                print(f"读取日志文件时出错: {e}")
        
        # 如果返回码非0且日志检查也失败
        print(f"\n错误: 微调脚本返回非零退出码: {result.returncode}")
        return False
        
    except subprocess.TimeoutExpired:
        print("错误: 微调脚本运行超时")
        return False
    except Exception as e:
        print(f"运行微调脚本时出错: {e}")
        return False

def main():
    """主函数：遍历所有预训练配置和 kshot 值"""
    parser = argparse.ArgumentParser(description='遍历微调组合（fedbook Baseline）')
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--seed", type=int, default=2025, help="随机种子")
    parser.add_argument("--config_dir", type=str, default=DEFAULT_CONFIG_DIR, 
                       help="预训练配置文件目录")
    parser.add_argument("--kshot_list", type=str, default=",".join(map(str, DEFAULT_KSHOT_LIST)),
                       help="kshot 值列表，用逗号分隔，例如: -1,1,5")
    parser.add_argument("--standard", type=int, default=DEFAULT_STANDARD,
                       help="每个实验运行的轮数")
    
    args = parser.parse_args()
    
    # 解析 kshot 列表
    try:
        kshot_list = [int(k.strip()) for k in args.kshot_list.split(",")]
    except ValueError:
        print(f"错误: 无法解析 kshot 列表 '{args.kshot_list}'")
        print("请使用逗号分隔的整数，例如: -1,1,5")
        return
    
    # 查找所有预训练配置文件
    config_files = find_pretrain_configs(args.config_dir)
    
    if len(config_files) == 0:
        print(f"错误: 在 {args.config_dir} 中未找到任何配置文件")
        return
    
    # 创建本次运行的日志子文件夹（带时间戳）
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_log_dir = osp.join(LOG_BASE_DIR, f"finetune_run_{timestamp}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    print("="*60)
    print("开始遍历微调组合（fedbook Baseline）")
    print(f"配置文件目录: {args.config_dir}")
    print(f"找到 {len(config_files)} 个预训练配置")
    print(f"K-shot 值: {kshot_list}")
    print(f"每个实验运行轮数: {args.standard}")
    print(f"总实验数: {len(config_files)} × {len(kshot_list)} = {len(config_files) * len(kshot_list)}")
    print(f"日志保存目录: {run_log_dir}")
    print("="*60)
    
    results = []
    total_experiments = len(config_files) * len(kshot_list)
    current_experiment = 0
    
    for config_idx, config_path in enumerate(config_files):
        config_name = osp.basename(config_path)
        
        # 检查模型是否存在
        model_exists, model_path = check_model_exists(config_path)
        if not model_exists:
            print(f"\n{'#'*60}")
            print(f"警告: 配置文件 {config_name} 对应的预训练模型不存在")
            print(f"预期模型路径: {model_path}")
            print(f"跳过该配置文件的所有微调实验")
            print(f"{'#'*60}\n")
            
            # 记录所有 kshot 的失败
            for kshot in kshot_list:
                current_experiment += 1
                results.append({
                    "config": config_name,
                    "kshot": kshot,
                    "success": False,
                    "reason": "预训练模型不存在"
                })
            continue
        
        print(f"\n{'#'*60}")
        print(f"配置文件 {config_idx+1}/{len(config_files)}: {config_name}")
        print(f"模型路径: {model_path}")
        print(f"{'#'*60}")
        
        for kshot_idx, kshot in enumerate(kshot_list):
            current_experiment += 1
            print(f"\n实验 {current_experiment}/{total_experiments}")
            print(f"K-shot: {kshot} ({kshot_idx+1}/{len(kshot_list)})")
            
            # 运行微调
            success = run_finetune(config_path, kshot, args.standard, run_log_dir,
                                 gpu_id=args.gpu_id, seed=args.seed)
            
            results.append({
                "config": config_name,
                "kshot": kshot,
                "success": success,
                "reason": "成功" if success else "运行失败"
            })
            
            if success:
                print(f"✓ 配置 {config_name}, K-shot {kshot} 微调完成")
            else:
                print(f"✗ 配置 {config_name}, K-shot {kshot} 微调失败")
    
    # 打印总结
    print("\n" + "="*60)
    print("微调组合运行总结")
    print("="*60)
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    print(f"总计: {len(results)} 个实验")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print("\n详细结果:")
    
    # 按配置分组显示结果
    current_config = None
    for result in results:
        if result["config"] != current_config:
            current_config = result["config"]
            print(f"\n配置文件: {current_config}")
        status = "✓ 成功" if result["success"] else f"✗ 失败 ({result['reason']})"
        print(f"  K-shot {result['kshot']}: {status}")
    
    print("\n" + "="*60)
    print(f"所有实验完成！")
    print(f"日志文件保存在: {run_log_dir}")
    print(f"汇总文件: {osp.join(run_log_dir, 'results_summary.log')}")
    print("="*60)

if __name__ == "__main__":
    main()

