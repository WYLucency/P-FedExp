import torch
import os.path as osp
import argparse
from flcore.ours.server import Server
from flcore.ours.client import Client
from utils.basic_utils import *
from utils.basic_utils import check_path, index_to_mask
from utils.partition_utils import graph_set_partition, single_graph_partition
import time
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--project_path", type=str, default=osp.join(osp.dirname(__file__)))
    parser.add_argument("--data_config", type=str, default=osp.join(osp.dirname(__file__), "config", "pretrain_config.json"))
    parser.add_argument("--partition_root", type=str, default=osp.join(osp.dirname(__file__), "partitions"))
    R1 = 20
    R2 = 20



    args = parser.parse_args()
    
    seed_everything(args.seed)
    if torch.cuda.is_available() and args.gpu_id is not None:
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.cuda.set_device(args.gpu_id)
    else:
        device = torch.device("cpu")
    
    # load global dataset
    num_clients_each_dataset, dataset_list = load_pretrain_mydataset_ofa(args)
    
    # Load num_experts from config
    with open(args.data_config, 'r', encoding='utf-8') as file:
        config_data = json.load(file)
        args.num_experts = config_data.get("num_experts", 4)
    
    
    local_data_list = []
    
    # 不同的数据集分配不同的任务
    for it, glb_data in enumerate(dataset_list):
        print(f"loading partition for {glb_data.name.lower()}...")
        # 【修改点1】在节点分类数据集列表中添加citeseer、photo、computers，支持这些数据集的图分区
        if glb_data.name.lower() in ["cora","pubmed","citeseer","wikics","arxiv","photo","computers"]:
            local_data_list += single_graph_partition(glb_data, num_partitions=num_clients_each_dataset[it], task="node_cls", root=args.partition_root)
        elif glb_data.name.lower() in ["wn18rr", "fb15k237"]:
            local_data_list += single_graph_partition(glb_data, num_partitions=num_clients_each_dataset[it], task="link_pre", root=args.partition_root)
        elif glb_data.name.lower() in ["chemhiv","chempcba", "chemblpre", "proteins", "imdb-binary"]:
            local_data_list += graph_set_partition(glb_data, num_partitions=num_clients_each_dataset[it], root=args.partition_root, mode="pretrain")
         
    # free memory for g/lobal datasets
    dataset_name_list = [i.name.lower() for i in dataset_list]
    del dataset_list
    

    
    
    num_clients = sum(num_clients_each_dataset)
    clients = []
    for client_id in range(num_clients):
        client = Client(args, client_id, local_data_list[client_id], device)
        clients.append(client)
        
    server = Server(args, device)
    
    
    
    # pre-training
    local_messages = {}


    for round_id in range(R1+R2): # 600 is ok?
        local_codebooks = []


        for client in clients:

            if round_id == 0:
                client.set_pretrain_model(server.get_global_message())
            else:
                client.set_pretrain_model(server.personalized_glb_model_list[client.client_id])


            # local_codebooks.append(client.vq._codebook) 
            # MoE does not use codebooks
            pass


        for client_id in range(num_clients):
            # client execute:
            print(f"Round {round_id}, Client {client_id}")
            clients[client_id].pretrain()
            local_messages[f"client_{client_id}"] = clients[client_id].get_pretrain_model()

        server.execute(local_messages, get_gfm=False)
  ###OFA要做的
        # # server execute:
        # if round_id <= R1:
        #     server.execute(local_messages, get_gfm=True)
        # else:
        #     server.execute(local_messages, get_gfm=False)
    
    
    
    
    
    # save models
    # 【FedBook版本】添加_FedBook后缀到模型路径
    model_path_base = "_".join([f"{dataset_name}_{num_clients_each_dataset[i]}" for i, dataset_name in enumerate(dataset_name_list)])
    model_path = osp.join(osp.dirname(__file__), 'ckpts', 'pretrain_models', f"{model_path_base}_fedbook")
    for client_id in range(num_clients):
        client_save_path = osp.join(model_path, f"client_{client_id}")
        check_path(client_save_path)
        clients[client_id].pretrain_model.save_encoder(osp.join(client_save_path, f"moe_encoder.pt"))
        clients[client_id].pretrain_model.save_gating(osp.join(client_save_path, f"gating.pt"))
        print("Client Save the model")
    server_save_path = osp.join(model_path, "server")
    check_path(server_save_path)
    server.global_model.save_encoder(osp.join(server_save_path, f"moe_encoder.pt"))
    server.global_model.save_gating(osp.join(server_save_path, f"gating.pt"))
    print("Server Save the model")