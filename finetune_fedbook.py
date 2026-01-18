import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
import argparse
from utils.basic_utils import *
from flcore.ours.server import Server
from flcore.ours.client import Client
import yaml
from utils.partition_utils import graph_set_partition, single_graph_partition


import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import time
import logging

# 日志保存配置
LOG_DIR = osp.join(osp.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

AUTO_SAVE_CONFIG = {
    "summary_file": osp.join(LOG_DIR, "results_summary.log")
}


class Logger:
    """同时输出到控制台和日志文件的Logger"""
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
    def close(self):
        self.log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--data_config", type=str, default=osp.join(osp.dirname(__file__), 'config', 'pretrain_config.json'))
    parser.add_argument("--partition_root", type=str, default=osp.join(osp.dirname(__file__), 'partitions'))
    parser.add_argument("--k_shot", type=int, default=-1)
    parser.add_argument("--standard", type=int, default=1,help="Number of runs for statistics (Mean & Std)")
    parser.add_argument("--log_dir", type=str, default=None, help="日志保存目录（如果指定，将使用该目录；否则使用默认logs目录）")
    
    args = parser.parse_args()
    
    # 确定日志目录
    if args.log_dir is not None:
        log_base_dir = args.log_dir
    else:
        log_base_dir = LOG_DIR
    
    os.makedirs(log_base_dir, exist_ok=True)
    
    # 设置日志文件（按时间戳命名，包含参数信息）
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    # log_filename = f"finetune_seed{args.seed}_kshot{args.k_shot}_{timestamp}.log"
    log_filename = f"FedBook_setting_kshot{args.k_shot}_standard{args.standard}_{timestamp}_graph.log"
    log_file_path = osp.join(log_base_dir, log_filename)
    
    # 重定向输出到日志文件
    sys.stdout = Logger(log_file_path)
    sys.stderr = Logger(log_file_path)
    
    print(f"=" * 60)
    print(f"Finetune started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_file_path}")
    print(f"Arguments: {vars(args)}")
    print(f"=" * 60)
    
    seed_everything(args.seed)
    if torch.cuda.is_available() and args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")

    # 汇总文件也保存在相同的日志目录中
    summary_file = osp.join(log_base_dir, "results_summary.log")
    
    
    # Load num_experts from config
    with open(args.data_config, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
        args.num_experts = config_data.get("num_experts", 4)

    num_clients_each_dataset, dataset_list = load_pretrain_mydataset_ofa(args, mode="finetune")

        
    local_data_list = []
    pretrain_list = []
    

    
    
    for it, glb_data in enumerate(dataset_list):
        print(f"loading partition for {glb_data.name.lower()}...")
        # 在节点分类数据集列表中添加citeseer、photo、computers、reddit，支持这些数据集的微调
        if glb_data.name.lower() in ["cora","pubmed","citeseer","wikics","arxiv","photo","computers","reddit"]:
            local_data_list += single_graph_partition(glb_data, num_partitions=num_clients_each_dataset[it], task="node_cls", root=args.partition_root)
        elif glb_data.name.lower() in ["wn18rr", "fb15k237"]:
            local_data_list += single_graph_partition(glb_data, num_partitions=num_clients_each_dataset[it], task="link_pre", root=args.partition_root)
        elif glb_data.name.lower() in ["chemhiv","chempcba", "chemblpre", "proteins", "imdb-binary"]:
            local_data_list += graph_set_partition(glb_data, num_partitions=num_clients_each_dataset[it], root=args.partition_root, mode="finetune")
    
    
    
    # free memory for global datasets
    del dataset_list

    # load finetuning params for each dataset
    # load finetuning params for each dataset
    with open(osp.join(osp.dirname(__file__), 'config', 'finetune.yaml'), 'r') as file:
        params = yaml.safe_load(file)
    
    params_list = []    
    for idx, data_tag in enumerate(local_data_list):
        # 统一用小写比较，避免数据集名称大小写不一致导致的问题
        dataset_name_lower = data_tag.name.lower()
        if dataset_name_lower in ["cora", "pubmed", "citeseer", "arxiv", "wikics", "photo", "computers", "reddit"]:
            client_params = params["node"][dataset_name_lower]
        elif dataset_name_lower in ["fb15k237", "wn18rr"]:
            client_params = params["link"][data_tag.name.upper()]
        else:
            client_params = params["graph"][data_tag.name]
        params_list.append(client_params)
                

    # load partitions
    num_clients = sum(num_clients_each_dataset)
    clients = []
    for client_id in range(num_clients):
        client = Client(args, client_id, local_data_list[client_id], device, params_list[client_id], mode="finetune")
        clients.append(client)
    server = Server(args, device)
    
    
    
    # load pretrained model
    # 加载预训练模型配置，支持跨域微调场景
    with open(args.data_config, 'r', encoding='utf-8') as file:
        dataset_config_pretrain = json.load(file)
        num_clients_each_dataset_pretrain = dataset_config_pretrain["pretrain_num_clients"]
        dataset_name_list_pretrain = dataset_config_pretrain["datasets"]
        # 【FedBook版本】添加_FedBook后缀到模型路径
        model_path_base = "_".join([f"{dataset.lower()}_{num_clients_each_dataset_pretrain[i]}" for i, dataset in enumerate(dataset_name_list_pretrain)])
        model_path = osp.join(osp.dirname(__file__), 'ckpts', 'pretrain_models', f"{model_path_base}_fedbook")
    
    # 跨域微调：所有微调客户端加载聚合后的服务器模型
    # 因为预训练和微调的客户端数量可能不同（如预训练3个，微调5个）
    # 所以统一使用服务器聚合后的全局模型
    print(f"Loading pretrained model from: {model_path}")
    print(f"Pretrain: {len(dataset_name_list_pretrain)} datasets, Finetune: {num_clients} clients")
    
    for client_id in range(num_clients):
        clients[client_id].moe_encoder.load_state_dict(torch.load(os.path.join(model_path, f'server/moe_encoder.pt')))
        clients[client_id].gating.load_state_dict(torch.load(os.path.join(model_path, f'server/gating.pt')))
        # clients[client_id].encoder.load_state_dict(torch.load(os.path.join(model_path, f'server/encoder.pt')))
        # clients[client_id].vq.load_state_dict(torch.load(os.path.join(model_path, f'server/vq.pt')))
        
        # clients[client_id].encoder.load_state_dict(torch.load(os.path.join(model_path, f'client_{client_id}/encoder.pt')))
        # clients[client_id].vq.load_state_dict(torch.load(os.path.join(model_path, f'client_{client_id}/vq.pt')))
    
    # final results
    result = {}
    NUM_RUNS = args.standard # 定义微调运行次数，保持前后一致
    
    # isolated finetune
    for client_id in range(num_clients):
        clients[client_id].data_tag = get_k_shot(clients[client_id].data_tag, k=args.k_shot)
        
        name = clients[client_id].data_tag.name.lower()

        # 运行 NUM_RUNS 次微调
        clients[client_id].finetune(standard=NUM_RUNS)

        if name not in result:
            result[name] = {}
            
        # 修改：循环范围从 range(1) 改为 range(NUM_RUNS)，获取所有 Run 的结果
        for standard_id in range(NUM_RUNS):
            if standard_id not in result[name]:
                result[name][standard_id] = []
            
            # 确保 logger 中有对应 standard_id 的记录
            if standard_id in clients[client_id].logger.best:
                single_best = clients[client_id].logger.best[standard_id]
                num_samples = clients[client_id].data_tag.test_mask.nonzero().shape[0]
                result[name][standard_id].append((single_best, num_samples))
            
        # 动态打印中间结果
        standard_dict = result[name]
        standard_result = []
        
        for values in standard_dict.values():
            if len(values) > 0:
                # 计算该 Run (standard_id) 下所有 Client 的加权平均
                result_test = sum([pair[0]['test'] * pair[1] for pair in values]) / sum([pair[1] for pair in values])
                standard_result.append(result_test)
        
        # 修改：在中间日志中也显示 STD
        if len(standard_result) > 0:
            curr_mean = np.mean(standard_result)
            curr_std = np.std(standard_result)
            _line = f"  [{name}] Client {client_id} done, current: {curr_mean:4f} ± {curr_std:.4f}%"
            print(_line)

    # 最终结果汇总
    final_results = {}
    for name, standards in result.items():
        dataset_scores = [] # 存储该数据集每一次 Run 的最终加权分数
        
        for std_id, values in standards.items():
            if values:  # 防止空列表
                weighted_scores = [pair[0]['test'] * pair[1] for pair in values]
                weights = [pair[1] for pair in values]
                mean_score = sum(weighted_scores) / sum(weights)
                dataset_scores.append(mean_score ) # 转为百分比
        
        # 修改：计算 Mean 和 STD
        if dataset_scores:
            final_mean = np.mean(dataset_scores)
            final_std = np.std(dataset_scores)
            final_results[name] = {"mean": final_mean, "std": final_std}
    
    # 打印最终汇总
    print(f"\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    for name, stats in final_results.items():
        print(f"{name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # 保存到汇总日志文件
    with open(summary_file, "a") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        f.write(f"Seed: {args.seed}, K-shot: {args.k_shot}\n")
        for name, stats in final_results.items():
            # 修改：写入格式包含 STD
            f.write(f"{name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        f.write("-" * 50 + "\n")
    
    print(f"=" * 60)
    print(f"Finetune completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results summary saved to: {summary_file}")
    print(f"Full log saved to: {log_file_path}")
    print(f"=" * 60)
    
    # 关闭日志记录器
    try:
        if isinstance(sys.stdout, Logger):
            sys.stdout.close()
            sys.stdout = sys.stdout.terminal
        if isinstance(sys.stderr, Logger):
            sys.stderr.close()
            sys.stderr = sys.stderr.terminal
    except Exception as e:
        # 即使关闭日志时出错，也不影响程序正常退出
        print(f"关闭日志记录器时出错（可忽略）: {e}")
    
    # 确保正常退出
    sys.exit(0)