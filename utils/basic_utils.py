import json
# from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import os.path as osp
import random
import numpy as np
from pathlib import Path
import os
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, DataLoader
import torch.nn.functional as F
import torch
from torchmetrics import Accuracy, AUROC
from sklearn.metrics import f1_score, roc_auc_score

def get_k_shot(data, k=-1):
    """
    对不同任务进行 k-shot 采样。
    
    支持：
    - 节点分类任务（Node Classification）
    - 图分类任务（Graph Classification）
    不支持链路预测任务（Link Prediction）。
    """
    if k == -1:
        return data
    
    # 检查是否为链路预测任务，如果是则跳过 k-shot 采样
    if hasattr(data, 'task') and 'link' in str(data.task).lower():
        return data
    
    # 检查数据集名称，WN18RR 和 FB15K237 是链路预测数据集
    if hasattr(data, 'name') and data.name.lower() in ['wn18rr', 'fb15k237']:
        return data
    
    # ===== 图分类任务处理 =====
    # 图分类数据集的特征：有 train_mask, val_mask, test_mask 作为图级别的掩码
    # 且 data 是一个 dataset（可通过 len() 和索引访问）
    if hasattr(data, 'name') and data.name.lower() in ['chemhiv', 'chempcba', 'chemblpre', 'proteins', 'imdb-binary']:
        # 获取所有训练图的索引
        train_indices = data.train_mask.nonzero(as_tuple=True)[0].tolist()
        
        # 获取训练集中的所有标签
        if hasattr(data, 'y'):
            # 对于图分类，y 可能是整个数据集的标签
            train_labels = data.y[data.train_mask]
        else:
            # 如果没有统一的 y，需要从每个图中获取
            train_labels = torch.tensor([data[i].y for i in train_indices])
        
        # 处理多任务标签（如 chempcba 有 128 个任务）
        if train_labels.dim() > 1:
            # 多任务二分类：对每个任务独立进行 k-shot 采样
            num_tasks = train_labels.shape[1]
            selected_indices_set = set()
            
            for task_idx in range(num_tasks):
                task_labels = train_labels[:, task_idx]
                
                # 对于每个类别（0和1）选择 k 个样本
                for class_label in [0, 1]:
                    # 找到该任务中该类别的所有样本索引
                    class_mask = (task_labels == class_label)
                    class_indices = [train_indices[i] for i, m in enumerate(class_mask) if m]
                    
                    # 选择前 k 个（如果样本数少于 k，则选择所有）
                    if k == 0:
                        selected = []
                    else:
                        selected = class_indices[:min(k, len(class_indices))]
                    selected_indices_set.update(selected)
            
            # 更新 train_mask
            new_train_mask = torch.zeros_like(data.train_mask)
            for idx in selected_indices_set:
                new_train_mask[idx] = True
                
        else:
            # 单任务分类
            unique_labels = train_labels.unique().tolist()
            new_train_mask = torch.zeros_like(data.train_mask)
            
            for label in unique_labels:
                # 找到该类别的所有训练样本索引
                label_mask = (train_labels == label)
                cls_train_idx = [train_indices[i] for i, m in enumerate(label_mask) if m]
                
                # 选择前 k 个样本
                if k == 0:
                    selected = []
                else:
                    selected = cls_train_idx[:min(k, len(cls_train_idx))]
                
                for idx in selected:
                    new_train_mask[idx] = True
        
        data.train_mask = new_train_mask
        print(f"[K-shot for Graph Classification] Dataset: {data.name}, k={k}, "
              f"Original training samples: {len(train_indices)}, "
              f"After k-shot: {new_train_mask.sum().item()}")
        return data
    
    # ===== 节点分类任务处理 =====
    # 检查 y 和 train_mask 的形状是否匹配
    if data.y.shape[0] != data.train_mask.shape[0]:
        return data

    new_train_mask = torch.zeros_like(data.train_mask)
    train_labels = data.y[data.train_mask]
    unique_labels = train_labels.unique().tolist()

    for label in unique_labels:
        cls_train_idx = ((data.y == label) & data.train_mask).nonzero(as_tuple=True)[0].tolist()
        if k == 0:
            selected = []
        else:
            selected = cls_train_idx[:k]
        new_train_mask[selected] = True

    data.train_mask = new_train_mask
    print(f"[K-shot for Node Classification] Dataset: {data.name if hasattr(data, 'name') else 'Unknown'}, k={k}, "
          f"Selected samples: {new_train_mask.sum().item()}")
    return data



from ofa.gp.utils.utils import (
    load_yaml,
    combine_dict,
    merge_mod,
    setup_exp,
    set_random_seed,
)
from utils.data_utils import (
    SentenceEncoder,
    MultiApr,
    MultiAuc,
    ENCODER_DIM_DICT,
)
from task.task_constructor import UnifiedTaskConstructor

metric2order = {'loss': 'min', 'acc': 'max', 'f1': 'max', 'precision': 'max', 'recall': 'max', 'auc': 'max',
                'ap': 'max', 'mcc': 'max', 'hit': 'max', 'ndcg': 'max', 'map': 'max', 'mrr': 'max'}
task_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__),"..", "config", "task_config.yaml"))
data_config_lookup = load_yaml(os.path.join(os.path.dirname(__file__),"..", "config", "data_config.yaml"))

class EarlyStopping:
    def __init__(self, patience=50):
        self.patience = patience
        self.counter = 0
        self.best_val = -np.inf
        self.best_dict = None
        self.early_stop = False

    def __call__(self, result):
        if result['val'] > self.best_val:
            self.best_val = result['val']
            self.best_dict = result
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

class Logger:
    def __init__(self):
        self.data = {}
        self.best = {}

    def check_result(self, result):
        if 'metric' not in result:
            raise ValueError('Result must contain metric key')
        if result['metric'] not in metric2order:
            raise ValueError('Metric not supported')
        if result['train'] is None:
            result['train'] = 0
        if result['val'] is None:
            result['val'] = 0

        return result

    def log(self, run, epoch, loss, result):
        result = self.check_result(result)

        train_value = result['train']
        val_value = result['val']
        test_value = result['test']

        if run not in self.data:
            self.data[run] = {'train': [], 'val': [], 'test': []}

        self.data[run]['loss_train'] = loss
        self.data[run]['train'].append(train_value)
        self.data[run]['val'].append(val_value)
        self.data[run]['test'].append(test_value)
        self.data[run]['epoch'] = epoch

        if run not in self.best:
            self.best[run] = {'train': None, 'val': None, 'test': None}

        if metric2order[result['metric']] == 'max':
            if self.best[run]['val'] is None or val_value >= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['epoch'] = epoch
        else:
            if self.best[run]['val'] is None or val_value <= self.best[run]['val']:
                self.best[run]['train'] = train_value
                self.best[run]['val'] = val_value
                self.best[run]['test'] = test_value
                self.best[run]['epoch'] = epoch

    def get_run_raw(self):
        return self.data

    def get_best_raw(self):
        return self.best

    def get_single_run(self, run_idx):
        return self.data[run_idx]

    def get_single_best(self, run_idx):
        return self.best[run_idx]

    def get_run(self):
        train = np.mean([np.mean(self.data[run_idx]['train']) for run_idx in self.data])
        val = np.mean([np.mean(self.data[run_idx]['val']) for run_idx in self.data])
        test = np.mean([np.mean(self.data[run_idx]['test']) for run_idx in self.data])
        return {'train': train, 'val': val, 'test': test}

    def get_best(self):
        train = [self.best[run_idx]['train'] for run_idx in self.best]
        val = [self.best[run_idx]['val'] for run_idx in self.best]
        test = [self.best[run_idx]['test'] for run_idx in self.best]

        return {'train': {'mean': np.mean(train), 'std': np.std(train)},
                'val': {'mean': np.mean(val), 'std': np.std(val)},
                'test': {'mean': np.mean(test), 'std': np.std(test)}}

def get_loader(data, split, labels, task, batch_size):
    task = task
    setting = "standard"

    if task == "node_cls":
        if setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            train_loader = NeighborLoader(
                data,
                num_neighbors=[10] * 2,
                input_nodes=mask2idx(split["train"]),
                batch_size=batch_size,
                shuffle=True,
            )
        subgraph_loader = NeighborLoader(
            data,
            num_neighbors=[-1] * 2,
            batch_size=512,
            shuffle=False,
        )
        return train_loader, subgraph_loader

    elif task == "link_pre":
        if setting in ['zero_shot', 'in_context']:
            train_loader = None
        else:
            train_loader = LinkNeighborLoader(
                data,
                num_neighbors=[30] * 2,
                edge_label_index=data.edge_index[:, split["train"]],
                edge_label=labels[split["train"]],
                batch_size=batch_size,
                shuffle=True,
            )
        subgraph_loader = LinkNeighborLoader(
            data,
            num_neighbors=[-1] * 2,
            edge_label_index=data.edge_index,
            edge_label=labels,
            batch_size=4096,
            shuffle=False,
        )
        return train_loader, subgraph_loader

    elif task == "graph_cls":
        if setting == 'standard':
            train_dataset = data[split["train"]]
            val_dataset = data[split["valid"]]
            test_dataset = data[split["test"]]

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
            )
        elif setting in ['few_shot']:
            # As we only update the train_idx in sampling few-shot samples,
            # we can directly use the split["train"] as the train_idx
            # This enables the shuffle function in DataLoader.
            # The drawback is we should define the proto_loader in the finetune_graph_task function
            train_dataset = data[split["train"]]

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
            )
            val_loader = None
            test_loader = None

        elif setting in ['zero_shot', 'in_context']:
            train_loader = None
            val_loader = None
            test_loader = None

        return train_loader, val_loader, test_loader


    
def seed_everything(seed):
    """
    Sets the seed for multiple random number generators to ensure reproducibility across runs. 
    It also configures the behavior of the CUDA backend for deterministic output.

    Args:
        seed (int): The seed number to use for seeding the random number generators.

    Details:
        - Sets the seed for Python's built-in `random` module, NumPy's random module, and PyTorch.
        - Configures PyTorch's CUDA-related seeds for all GPUs.
        - Sets CUDA's cuDNN backend to operate deterministically, which can impact performance
          due to the disabling of certain optimizations like `benchmark` and general `enabled` status.

    Note:
        Enabling determinism can lead to a performance trade-off but is necessary for reproducibility
        when exact outcomes are critical to maintain across different runs, especially during debugging
        or testing phases.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
   

def load_finetune_mydataset_ofa(args):
    with open(args.data_config, 'r', encoding='utf-8') as file:
        dataset_config = json.load(file)
        num_clients = dataset_config["finetune_num_clients"]
        dataset_list = []
        split_list = []
        labels_list = []
        for dataset_name in dataset_config["finetune_datasets"]:
            tasks = get_task_constructor('/home/zhuangzy25/FGLFGM/FedBook_supp/archive/cache_data')
            # 统一转换为小写进行比较
            dataset_name_lower = dataset_name.lower()
            if dataset_name_lower in ["cora", "pubmed", "wikics", "arxiv", "photo", "computers"]:
                dataset, split, labels, num_classes, num_tasks = get_node_data(tasks, dataset_name_lower)
            elif dataset_name in ["WN18RR", "FB15K237"]:
                dataset, split, labels, num_classes, num_tasks = get_link_data(tasks, dataset_name)
            elif dataset_name_lower in ["chemhiv", "chemblpre", "chempcba", "proteins", "imdb-binary"]:
                dataset, split, labels, num_classes, num_tasks = get_graph_clf_graph(tasks, dataset_name_lower)
            dataset_list.append(dataset)
            split_list.append(split)
            labels_list.append(labels)
        task_list = dataset_config["tasks"]
        weight_list = dataset_config["weights"]
    return num_clients, dataset_list, task_list, weight_list, split_list, labels_list

def refine_dataset(dataset):
    # works for molecule graphs
    if dataset.data.get("node_embs") is not None:
        dataset.data.node_text_feat = dataset.data.node_embs
        # dataset.data.node_embs = None
    if dataset.data.get("edge_embs") is not None:
        dataset.data.edge_text_feat = dataset.data.edge_embs
        # dataset.data.edge_embs = None
    if dataset.data.get("pretrain_edge_index") is not None:
        dataset.data.edge_index = dataset.data.pretrain_edge_index
        # dataset.data.pretrain_edge_index = None
    return dataset

def load_pretrain_mydataset_ofa(args, mode="pretrain"):
    with open(args.data_config, 'r', encoding='utf-8') as file:
        dataset_config = json.load(file)
        task_list = dataset_config["tasks"]
        weight_list = dataset_config["weights"]
        
        
        if mode == "pretrain":
            num_clients_each_dataset = dataset_config["pretrain_num_clients"] 
            datasets = dataset_config["datasets"]
        else:
            num_clients_each_dataset = dataset_config["finetune_num_clients"]
            datasets = dataset_config["finetune_datasets"]
            
        dataset_list = []
        split_list = []

        tasks = get_task_constructor(osp.join(args.project_path, "cache_data"))
        
        for dataset_name in datasets:
            # 【修改点3】在节点分类数据集列表中添加citeseer、photo、computers、reddit，支持这些数据集的加载
            # 统一转换为小写进行比较
            dataset_name_lower = dataset_name.lower()
            if dataset_name_lower in ["cora", "pubmed", "citeseer", "wikics", "arxiv", "photo", "computers", "reddit"]:
                dataset, split, labels, num_classes, num_tasks = get_node_data(tasks, dataset_name_lower)
                dataset = filter_unnecessary_attrs(dataset)
            elif dataset_name in ["WN18RR", "FB15K237"]:
                dataset, split, labels, num_classes, num_tasks = get_link_data(tasks, dataset_name)
                dataset = filter_unnecessary_attrs(dataset)
            elif dataset_name_lower in ["chemhiv"]:
                dataset, split, labels, num_classes, num_tasks = get_graph_clf_graph(tasks, dataset_name_lower)
            elif dataset_name_lower in ["chempcba"]:
                dataset, split, labels, num_classes, num_tasks = get_graph_clf_graph(tasks, dataset_name_lower)
            elif dataset_name_lower in ["proteins", "imdb-binary"]:
                dataset, split, labels, num_classes, num_tasks = get_graph_clf_graph(tasks, dataset_name_lower)
            elif dataset_name_lower in ["chemblpre"]:
                data_config = data_config_lookup[dataset_name_lower]
                dataset = tasks.get_ofa_data(data_config)
                labels = dataset.y
                split = {
                    'train': list(range(len(dataset))),'valid': [],'test': []
                }
                num_tasks = 1295
                num_classes = None
            else:
                raise ValueError(f"Unknown dataset: {dataset_name}. Please check your configuration.")
            
            
            dataset.num_tasks = num_tasks
            dataset.y = labels
            dataset = span_node_and_edge_idx(dataset)
            dataset.node_text = dataset.texts[0]
            dataset.edge_text = dataset.texts[1]
            
            
            dataset_list.append(dataset)
            split_list.append(split)
            

    
    for idx, data_tag in enumerate(dataset_list):
        data_tag.task = task_list[idx]
        data_tag.weight = weight_list[idx]
   
        
    return num_clients_each_dataset, dataset_list




def filter_unnecessary_attrs(dataset, mode="pretrain"):
    keys = [
        "x",
        "xe",
        "edge_index",
        "node_text_feat",
        "edge_text_feat",
        "class_node_text_feat",
    ]

    if mode == 'pretrain':
        keys = [
            "x",
            "xe",
            "edge_index",
            "node_text_feat",
            "edge_text_feat",
        ]

    if hasattr(dataset, "data"):
        for k, v in dataset.data.to_dict().items():
            if k not in keys:
                dataset.data[k] = None
                
    else:
        for k, v in dataset.to_dict().items():
            if k not in keys:
                dataset[k] = None
    return dataset


def span_node_and_edge_idx(dataset):
    if hasattr(dataset, "data"):
        # Define node index
        if dataset.data.x.ndim == 1:
            return dataset

        num_nodes = dataset.data.x.shape[0]
        dataset.data.x = torch.arange(num_nodes)

        # Define edge index
        num_edge_types = dataset.data.edge_text_feat.shape[0]
        num_edges = dataset.data.edge_index.shape[1]

        if num_edge_types == 1:
            dataset.data.xe = torch.zeros([num_edges], dtype=torch.long)
        else:
            dataset.data.xe = dataset.data.edge_types
    else: # local
        # Define node index
        if dataset.x.ndim == 1:
            return dataset

        num_nodes = dataset.x.shape[0]
        dataset.x = torch.arange(num_nodes)

        # Define edge index
        num_edge_types = dataset.edge_text_feat.shape[0]
        num_edges = dataset.edge_index.shape[1]

        if num_edge_types == 1:
            dataset.xe = torch.zeros([num_edges], dtype=torch.long)
        else:
            dataset.xe = dataset.edge_types
            
    return dataset

def pre_node(dataset):
    dataset = span_node_and_edge_idx(dataset)
    # dataset = filter_unnecessary_attrs(dataset)
    return dataset


def pre_link(dataset):
    dataset = span_node_and_edge_idx(dataset)
    # dataset = filter_unnecessary_attrs(dataset, mode="finetune")
    return dataset


def pre_graph(dataset):
    return dataset

def get_preprocess(task):
    if task == 'node_cls':
        return pre_node
    elif task == 'link_pre':
        return pre_link
    elif task == 'graph_cls':
        return pre_graph
    else:
        raise NotImplementedError('The task is not implemented')

def preprocess_split(split):
    if isinstance(split, dict):
        split_list = []
        if isinstance(split["test"], list):
            for train, valid, test in zip(split["train"], split["valid"], split["test"]):
                split_list.append({"train": train, "valid": valid, "test": test})
        elif split["test"].ndim == 1:
            for train, valid in zip(split["train"], split["valid"]):
                split_list.append({"train": train, "valid": valid, "test": split["test"]})
        return split_list
# def load_mini_lm(args):
#     with open(args.lm_config, 'r', encoding='utf-8') as file:
#         lm_config = json.load(file)
        
#     if lm_config["mini_lm"] == "sentence_bert":
#         from sentence_transformers import SentenceTransformer
#         mini_lm = SentenceTransformer(os.path.join(lm_config["root"], "multi-qa-distilbert-cos-v1"))
#     elif lm_config["mini_lm"] == "deberta":
#         from sentence_transformers import SentenceTransformer
#         mini_lm = SentenceTransformer(os.path.join(lm_config["root"], "deberta-v3-base"))
#     elif lm_config["mini_lm"] == "roberta":
#         from sentence_transformers import SentenceTransformer
#         mini_lm = SentenceTransformer(os.path.join(lm_config["root"], "all-MiniLM-L6-v2"))
#     elif lm_config["mini_lm"] == "tf-idf":
#         from model.non_param_lang_model import TFIDFModel
#         mini_lm = TFIDFModel()
#     elif lm_config["mini_lm"] == "word2vec":
#         from model.non_param_lang_model import Word2VecModel
#         mini_lm = Word2VecModel()
#     return mini_lm

def load_llm(args):
    with open(args.lm_config, 'r', encoding='utf-8') as file:
        lm_config = json.load(file)
    
    if lm_config["llm"] == "llama2-7b":        
        llm_tokenizer = LlamaTokenizer.from_pretrained(os.path.join(lm_config["root"], "Llama-2-7b-hf"), add_eos_token=True, padding_side="left")
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model = LlamaForCausalLM.from_pretrained(os.path.join(lm_config["root"], "Llama-2-7b-hf"))
    elif lm_config["llm"] == "e5-large-v2":
        from transformers import AutoModel, AutoTokenizer
        llm_tokenizer = AutoTokenizer.from_pretrained(os.path.join(lm_config["root"], "e5-large-v2"), add_eos_token=True, padding_side="left")
        llm_tokenizer.add_special_tokens({"pad_token": "<pad>"})
        llm_model = AutoModel.from_pretrained(os.path.join(lm_config["root"], "e5-large-v2"))
    elif lm_config["llm"] == "llama2-13b":
        llm_tokenizer = LlamaTokenizer.from_pretrained(os.path.join(lm_config["root"], "Llama-2-13b-hf"), add_eos_token=True, padding_side="left")
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
        llm_model = LlamaForCausalLM.from_pretrained(os.path.join(lm_config["root"], "Llama-2-13b-hf"))
        
    return llm_tokenizer, llm_model

        
    
    
        
    
def check_path(path):
    if not osp.exists(path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    return path    
    
def accuracy(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    return torch.sum(y_pred == y_true).item() / len(y_true)

def load_lm(args):
    with open(args.lm_config, 'r', encoding='utf-8') as file:
        lm_config = json.load(file)

def mask2idx(mask):
    return torch.where(mask == True)[0]

def get_device_from_model(model):
    return next(model.parameters()).device
def sample_proto_instances(labels, split, num_instances_per_class=10):
    y = labels.cpu().numpy()
    target_y = y[split.detach().cpu()]
    classes = np.unique(target_y)

    class_index = []
    for i in classes:
        c_i = np.where(y == i)[0]
        c_i = np.intersect1d(c_i, split)
        class_index.append(c_i)

    proto_idx = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        proto_idx = np.concatenate((proto_idx, idx[:num_instances_per_class]))

    return proto_idx.astype(int)
task2metric = {'node_cls': 'acc', 'link_pre': 'acc', 'graph_cls': 'auc'}
def evaluate(pred, y, task, mask=None, params=None):
    if mask is not None and mask.sum() == 0:
        return -999
    
    metric = task2metric[task]

    if metric == 'acc':
        return eval_acc(pred, y, mask) * 100
    elif metric == 'auc':
        return eval_auc(pred, y) * 100
    else:
        raise ValueError(f"Metric {metric} is not supported.")


def eval_acc(y_pred, y_true, mask):
    device = y_pred.device
    num_classes = y_pred.size(1)

    evaluator = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    if mask is not None:
        return evaluator(y_pred[mask], y_true[mask]).item()
    else:
        return evaluator(y_pred, y_true).item()


def eval_auc(y_pred, y_true):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    # 处理 1D 输入（单任务二分类）
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    # 确保 y_pred 和 y_true 的第二维匹配
    if y_pred.shape[1] != y_true.shape[1]:
        min_tasks = min(y_pred.shape[1], y_true.shape[1])
        y_pred = y_pred[:, :min_tasks]
        y_true = y_true[:, :min_tasks]

    roc_list = []
    y_true[y_true == -1] = 0
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_valid = y_true[:, i] == y_true[:, i]
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))

    # if len(roc_list) < y_true.shape[1]:
    #     print("Some target is missing!")
    #     print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    if len(roc_list) == 0:
        # 如果没有有效的 AUC 计算，返回 0.5（随机猜测）
        return 0.5
    
    return sum(roc_list) / len(roc_list)  # y_true.shape[1]

import torch


# def sample_proto_instances_for_graph(labels, split, num_instances_per_class=10):
#     y = labels
#     ndim = y.ndim
#     if ndim == 1:
#         y = y.reshape(-1, 1)

#     # Map class and instance indices

#     if isinstance(y, torch.Tensor):
#         y = y.cpu().numpy()
#     target_y = y[split]
#     task_list = target_y.shape[1]

#     # class_index_pos = {}
#     # class_index_neg = {}
#     task_index_pos, task_index_neg = [], []
#     for i in range(task_list):
#         c_i = np.where(y[:, i] == 1)[0]
#         c_i = np.intersect1d(c_i, split)
#         task_index_pos.append(c_i)

#         c_i = np.where(y[:, i] == 0)[0]
#         c_i = np.intersect1d(c_i, split)
#         task_index_neg.append(c_i)

#     assert len(task_index_pos) == len(task_index_neg)

#     # Randomly select instances for each task

#     proto_idx, proto_labels = {}, {}
#     for task, (idx_pos, idx_neg) in enumerate(zip(task_index_pos, task_index_neg)):
#         tmp_proto_idx, tmp_labels = np.array([]), np.array([])

#         # Randomly select instance for the task

#         np.random.shuffle(idx_pos)
#         np.random.shuffle(idx_neg)
#         idx_pos = idx_pos[:num_instances_per_class]
#         idx_neg = idx_neg[:num_instances_per_class]

#         # Store the randomly selected instances

#         tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_pos))
#         tmp_labels = np.concatenate((tmp_labels, np.ones(len(idx_pos))))
#         tmp_proto_idx = np.concatenate((tmp_proto_idx, idx_neg))
#         tmp_labels = np.concatenate((tmp_labels, np.zeros(len(idx_neg))))

#         proto_idx[task] = tmp_proto_idx.astype(int)
#         proto_labels[task] = tmp_labels.astype(int)

#     return proto_idx, proto_labels

def sample_proto_instances_for_graph(labels, num_tasks, split, num_instances_per_class=10):
    y = labels.view(-1, num_tasks)
    
    split_indices = torch.nonzero(split, as_tuple=True)[0]
    
    sampled_indices = {}
    
    for task in range(num_tasks):
        task_labels = y[:, task]
        

        task_sampled_indices = {}
        

        for class_label in [0, 1]:
            class_indices = split_indices[task_labels[split_indices] == class_label]
            if len(class_indices) > 0:
                sampled_class_indices = class_indices[torch.randperm(len(class_indices))[:num_instances_per_class]]
                task_sampled_indices[class_label] = sampled_class_indices
        
        if len(task_sampled_indices) != 0:
            sampled_indices[task] = task_sampled_indices
    
    return sampled_indices





llm_name = "ST"
def get_task_constructor(data_path):
    # Load processed_params.yaml
    encoder = SentenceEncoder(llm_name, root=data_path, batch_size=1)
    # 【修改点1】在task_names列表中添加citeseer、photo、computers、reddit相关任务，支持节点分类和链接预测
    # 【新增】添加 PROTEINS 和 IMDB-BINARY 图分类数据集
    task_names = ['cora_link', 'cora_node', 'pubmed_link', 'pubmed_node', 'citeseer_link', 'citeseer_node', 'photo', 'computers', 'arxiv', 'WN18RR', 'FB15K237', 'wikics', 'reddit', 'chemblpre', 'chempcba', 'chemhiv', 'proteins', 'imdb-binary']

    if isinstance(task_names, str):
        task_names = [a.strip() for a in task_names.split(",")]
    else:
        task_names = task_names

    root = data_path
    if llm_name != "ST":
        root = f"{data_path}_{llm_name}"

    tasks = UnifiedTaskConstructor(
        task_names,
        encoder,
        task_config_lookup,
        data_config_lookup,
        root=root,
        batch_size=512,
        sample_size=-1,
    )

    return tasks


def get_graph_clf_graph(tasks, dataset_name):
    data_config = data_config_lookup[dataset_name]
    dataset = tasks.get_ofa_data(data_config)
    split = tasks.get_data_split(data_config)

    if dataset_name in ["chemhiv"]:
        num_tasks = 1
        num_classes = None
        labels = dataset.y
    elif dataset_name in ["chempcba"]:
        num_tasks = 128
        num_classes = None
        labels = dataset.y.reshape(-1, num_tasks)
    elif dataset_name in ["chemblpre"]:
        raise NotImplementedError(f"Dataset {dataset_name} is only used for pre-training")
    elif dataset_name in ["proteins", "imdb-binary"]:
        # ========== 处理 TUDataset (PROTEINS, IMDB-BINARY) ==========
        # 这些数据集由 TUDatasetOFADataset 处理，已经包含必要的特征
        print(f"\n[TUDataset] 正在处理 {dataset_name} 数据集...")
        
        num_tasks = 1
        labels = dataset.y
        num_classes = labels.unique().shape[0]
        
        # 如果 split 为空，手动生成随机划分（Train/Val/Test = 80%/10%/10%）
        if split is None or (isinstance(split, dict) and len(split.get('train', [])) == 0):
            print(f"  警告: {dataset_name} 没有预定义划分，生成随机划分 (0.8/0.1/0.1)...")
            num_graphs = len(dataset)
            indices = list(range(num_graphs))
            np.random.shuffle(indices)
            
            train_end = int(0.8 * num_graphs)
            val_end = int(0.9 * num_graphs)
            
            split = {
                'train': indices[:train_end],
                'valid': indices[train_end:val_end],
                'test': indices[val_end:]
            }
        
        # TUDatasetOFADataset 已经在初始化时创建了所有必要的嵌入和文本
        # 检查是否已有必要属性，如果没有则说明有问题
        if not hasattr(dataset, 'node_embs') or not hasattr(dataset, 'edge_embs'):
            raise RuntimeError(
                f"TUDataset {dataset_name} 缺少必要的嵌入属性。"
                f"请确保 TUDatasetOFADataset 正确初始化。"
            )
        
        # 检查 texts 是否存在
        if not hasattr(dataset, 'texts'):
            # 如果 texts 在处理过程中丢失，重新添加
            dataset.texts = [["node"], ["edge"]]
        
        # 检查 class_node_text_feat 是否存在
        if not hasattr(dataset, 'class_node_text_feat'):
            # 如果丢失，重新创建
            target_dim = 768
            dataset.class_node_text_feat = torch.randn(num_classes, target_dim)
            print(f"  重新创建 class_node_text_feat: {dataset.class_node_text_feat.shape}")
        
        print(f"  ✓ TUDataset 处理完成:")
        print(f"    - node_embs: {dataset.data.node_embs.shape}")
        print(f"    - edge_embs: {dataset.data.edge_embs.shape}")
        print(f"    - texts: {len(dataset.texts[0])} 种节点, {len(dataset.texts[1])} 种边")
        print(f"    - num_classes: {num_classes}, num_graphs: {len(dataset)}\n")
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for graph classification task")

    return dataset, split, labels, num_classes, num_tasks

def get_node_data(tasks, dataset_name):
    data_config = data_config_lookup[dataset_name]
    dataset = tasks.get_ofa_data(data_config)
    data = dataset[0]

    num_tasks = 1

    # 【修改点2】在cora和pubmed的处理逻辑中添加citeseer、photo、computers、reddit，它们使用相同的CiteSplitter
    if dataset_name in ["cora", "pubmed", "citeseer", "photo", "computers", "reddit"]:
        # 【关键修复】Amazon数据集（Computers和Photo）可能没有train_masks属性，需要检查并生成
        if not hasattr(data, 'train_masks') or data.train_masks is None:
            print(f"警告: {dataset_name} 数据集缺少 train_masks 属性，正在生成随机划分...")
            # 为Amazon数据集生成随机划分（参考gen_data.py中的逻辑）
            num_nodes = data.num_nodes
            labels = data.y
            num_classes = labels.unique().shape[0]
            num_splits = 20  # 生成20个随机split
            
            train_masks = []
            val_masks = []
            test_masks = []
            
            # 设置随机种子以确保可重复性
            torch.manual_seed(42)
            
            for i in range(num_splits):
                train_mask = torch.zeros(num_nodes, dtype=torch.bool)
                val_mask = torch.zeros(num_nodes, dtype=torch.bool)
                test_mask = torch.zeros(num_nodes, dtype=torch.bool)
                
                # 为每个类别随机分配节点到train/val/test
                for class_id in range(num_classes):
                    class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
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
            
            # 将生成的masks保存到data对象中
            data.train_masks = train_masks
            data.val_masks = val_masks
            data.test_masks = test_masks
            print(f"已为 {dataset_name} 生成 {num_splits} 个随机划分")
        
        split = {"train": data.train_masks, "valid": data.val_masks, "test": data.test_masks}
        labels = data.y
        num_classes = labels.unique().shape[0]

    # elif dataset_name in ["wikics"]:
    #     split = {"train": data.train_mask.T, "valid": data.val_mask.T, "test": data.test_mask.T}
    #     labels = data.y
    #     num_classes = labels.unique().shape[0]

    # elif dataset_name in ["arxiv"]:
    #     split = {"train": data.train_mask, "valid": data.val_mask, "test": data.test_mask}
    #     labels = data.y.squeeze()
    #     num_classes = labels.unique().shape[0]
    elif dataset_name in ["wikics"]:
        # 定义一个简单的辅助函数，只有当维度大于1时才转置
        def safe_transpose(tensor):
            if tensor.dim() > 1:
                return tensor.transpose(0, 1)
            return tensor

        split = {
            "train": safe_transpose(data.train_mask),
            "valid": safe_transpose(data.val_mask),
            "test": safe_transpose(data.test_mask)
        }
        
        labels = data.y
        num_classes = labels.unique().shape[0]

    elif dataset_name in ["arxiv"]:
        # 修复 AttributeError: 'GlobalStorage' object has no attribute 'train_mask'
        # Arxiv (OGB) 通常使用索引(index)而不是掩码(mask)
        # 我们需要先获取索引，然后把它们转换成掩码，因为你的代码后续逻辑(mask2idx)需要掩码
        
        # 尝试从 dataset 获取切分索引
        if hasattr(dataset, 'get_idx_split'):
            split_idx = dataset.get_idx_split()
        else:
            # 如果 dataset 本身没有，尝试从 data 对象里找 (有些版本可能放在这里)
            # 或者直接报错提示
            print(f"警告: 无法在 Arxiv 数据集中找到 get_idx_split 方法，尝试直接读取属性。Available keys: {data.keys}")
            # 这里是一个备用方案，防止彻底崩溃，但通常上面那个 hasattr 应该为 True
            split_idx = {"train": getattr(data, 'train_mask', None), "valid": getattr(data, 'val_mask', None), "test": getattr(data, 'test_mask', None)}

        # 将索引转换为掩码 (index_to_mask 函数在你的 basic_utils.py 里已经定义了)
        num_nodes = data.num_nodes
        split = {
            "train": index_to_mask(split_idx["train"], num_nodes),
            "valid": index_to_mask(split_idx["valid"], num_nodes),
            "test": index_to_mask(split_idx["test"], num_nodes)
        }
        
        labels = data.y.squeeze()
        num_classes = labels.unique().shape[0]

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for node classification task")

    return dataset, split, labels, num_classes, num_tasks

def get_link_data(tasks, dataset_name):
    if dataset_name in ["WN18RR", "FB15K237"]:
        data_config = data_config_lookup[dataset_name]
        dataset = tasks.get_ofa_data(data_config)
        split = tasks.get_data_split(data_config)
        num_tasks = 1

        data = dataset[0]

        labels = data.edge_types
        num_classes = labels.unique().shape[0]

    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported for link classification task")

    return dataset, split, labels, num_classes, num_tasks
def index_to_mask(index_tensor, num_samples):

    mask_tensor = torch.zeros(num_samples, dtype=torch.bool)
    
    mask_tensor[index_tensor] = True
    
    return mask_tensor


def construct_graph(x):
    graph = [x, torch.tensor([]).view(2,-1).to(x.device).long(), None]
    return graph