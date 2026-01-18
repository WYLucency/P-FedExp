import numpy as np

import torch
from torch_geometric.loader import NeighborLoader
from utils.basic_utils import get_device_from_model, mask2idx, sample_proto_instances, evaluate, task2metric


def ft_node(model, dataset, loader, optimizer, split, labels, num_classes, no_proto_clf, no_lin_clf, use_z_in_predict, 
            query_node_code_first, lambda_proto, lambda_act, num_instances_per_class, scheduler=None, **kwargs):
    model.train()

    device = get_device_from_model(model)
    setting = "standard"
    num_classes = num_classes

    use_proto_clf = not no_proto_clf
    use_lin_clf = not no_lin_clf
    proto_loss = torch.tensor(0.0).to(device)
    act_loss = torch.tensor(0.0).to(device)

    mini_batch = loader is not None

    if not mini_batch:
        # Encode

        x = dataset.node_text_feat[dataset.x]
        edge_index = dataset.edge_index
        # edge_attr = dataset.edge_text_feat[dataset.edge_map]
        edge_attr = dataset.edge_text_feat[dataset.xe]
        y = labels.to(device)

        z = model.encode(x, edge_index, edge_attr)
        train_mask = split["train"]
        z_train, y_train = z[train_mask], y[train_mask]
    
        if use_proto_clf:
            # Compute Prototypes
            code_train, commit_loss = model.get_codes(z_train, use_orig_codes=True)

            proto_emb = model.get_class_prototypes(code_train, y_train, num_classes)
            if proto_emb is not None:
                proto_emb = proto_emb.detach()
                query_emb = z_train if use_z_in_predict else code_train

                # Compute Losses
                proto_loss = model.compute_proto_loss(query_emb, proto_emb, y_train) * lambda_proto

        if use_lin_clf:
            act_loss = model.compute_activation_loss(z_train, y_train) * lambda_act

        loss = proto_loss + act_loss

        optimizer.zero_grad()
        loss.backward()
        
        print(f"loss: {loss:.4f}")
        optimizer.step()
        if scheduler:
            scheduler.step()

        return {
            'proto_loss': proto_loss.item(),
            'act_loss': act_loss.item(),
            'loss': loss.item(),
        }
    else:
        if use_proto_clf:
            # Define Prototype Loader.

            # This is a unique step for mini-batch training
            # As we cannot use all instances to compute prototypes

            if setting == "standard":
                # You only need to sample instances for standard setting
                # In few-shot, we could use all instances available
                proto_idx = sample_proto_instances(
                    labels.cpu(),
                    mask2idx(split["train"].cpu()),
                    num_instances_per_class=num_instances_per_class,
                )
                # 【修复】当 num_instances_per_class=0 时，proto_idx 可能为空
                # 此时使用 train_loader 来计算原型，或者使用所有训练节点
                if len(proto_idx) == 0:
                    # 如果采样结果为空，使用 train_loader 本身来计算原型
                    proto_loader = loader
                else:
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=kwargs["num_neighbors"],
                        input_nodes=proto_idx,
                        batch_size=512,
                        num_workers=8,
                    )
            elif setting in ["few_shot"]:
                # In few-shot setting, we use the same train_loader for all tasks
                # As the number of few-shot instances is small enough
                proto_loader = loader

            # Compute Prototypes

            code_list, y_list = [], []
            for batch in proto_loader:
                batch = batch.to(device)
                bs = batch.batch_size

                x = batch.node_text_feat
                edge_index = batch.edge_index
                edge_attr = batch.edge_text_feat[batch.xe]

                y = batch.y[:bs]
                z = model.encode(x, edge_index, edge_attr)[:bs]

                code, _ = model.get_codes(z, use_orig_codes=True)
                code_list.append(code.detach())
                y_list.append(y)

            # 【修复】确保 code_list 不为空
            if len(code_list) == 0:
                raise ValueError(f"proto_loader 没有产生任何数据，无法计算原型。请检查 num_instances_per_class 设置或训练数据。")

            code = torch.cat(code_list, dim=0)
            y = torch.cat(y_list, dim=0)
            proto_emb = model.get_class_prototypes(code, y, num_classes)

        # Start Training

        total_proto_loss = 0
        total_act_loss = 0
        total_loss = 0

        for batch in loader:
            batch = batch.to(device)
            bs = batch.batch_size

            # Encode
            x = batch.node_text_feat
            edge_index = batch.edge_index
            edge_attr = batch.edge_text_feat[batch.xe]

            y = batch.y[:bs]
            z = model.encode(x, edge_index, edge_attr)[:bs]
            
            
            if use_proto_clf:
                code, commit_loss = model.get_codes(z, use_orig_codes=True)
                query_emb = z if use_z_in_predict else code
                proto_loss = model.compute_proto_loss(query_emb, proto_emb, y) * lambda_proto
            if use_lin_clf:
                act_loss = model.compute_activation_loss(z, y) * lambda_act

            loss = proto_loss + act_loss

            total_proto_loss += proto_loss.item()
            total_act_loss += act_loss.item()
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        return {
            'proto_loss': total_proto_loss / len(loader),
            'act_loss': total_act_loss / len(loader),
            'loss': total_loss / len(loader),
        } 
        
def eval_node(model, dataset, loader, split, labels, num_classes, no_proto_clf, no_lin_clf, use_z_in_predict,
              query_node_code_first, num_instances_per_class, task, **kwargs):
    model.eval()
    device = get_device_from_model(model)
    setting = "standard"
    num_classes = num_classes

    use_proto_clf = not no_proto_clf
    use_lin_clf = not no_lin_clf
    pred_proto = 0
    pred_lin = 0

    mini_batch = loader is not None
    if not mini_batch:
        # Encode

        x = dataset.node_text_feat[dataset.x]
        edge_index = dataset.edge_index
        # edge_attr = dataset.edge_text_feat[dataset.edge_map]
        edge_attr = dataset.edge_text_feat[dataset.xe]
        y = labels.to(x.device)

        z = model.encode(x, edge_index, edge_attr)

        if setting == "standard":

            if use_proto_clf:
                # Compute Prototypes
                train_mask = split["train"]
                code, _ = model.get_codes(z, use_orig_codes=True)
                code_train, y_train = code[train_mask], y[train_mask]

                proto_emb = model.get_class_prototypes(code_train, y_train, num_classes)
                if proto_emb is not None:
                    proto_emb = proto_emb.detach()
                    query_emb = z if model.use_z_in_predict else code

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                else:
                    pred_proto = torch.zeros(z.shape[0], num_classes, device=z.device)

            if use_lin_clf:
                pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
              

            pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

            # Evaluate
            train_mask, val_mask, test_mask = split["train"], split["valid"], split["test"]
            train_value = evaluate(pred, y, task,train_mask)
            val_value = evaluate(pred, y, task, val_mask)
            test_value = evaluate(pred, y, task, test_mask)

            return {
                'train': train_value,
                'val': val_value,
                'test': test_value,
                'metric': task2metric[task]
            }

        elif setting == "few_shot":
            n_task = len(split["valid"]["support"])
            train_values, val_values, test_values = [], [], []

            for i in range(n_task):
                s_mask = split["valid"]["support"][i]
                q_mask = split["valid"]["query"][i]

                if use_proto_clf:
                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_support, y_support = code[s_mask], y[s_mask]
                    z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]

                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes)
                    if proto_emb is not None:
                        proto_emb = proto_emb.detach()
                        query_emb = z_query if use_z_in_predict else code_query

                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                    else:
                        pred_proto = torch.zeros(z_query.shape[0], num_classes, device=z_query.device)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)
                  

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate

                value = evaluate(pred, y_query, task)
                train_values.append(value)
                val_values.append(value)

            for i in range(n_task):
                s_mask = split["test"]["support"][i]
                q_mask = split["test"]["query"][i]

                if use_proto_clf:
                    # Compute Prototypes

                    code, _ = model.get_codes(z, use_orig_codes=True)
                    code_support, y_support = code[s_mask], y[s_mask]
                    z_query, code_query, y_query = z[q_mask], code[q_mask], y[q_mask]

                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes)
                    if proto_emb is not None:
                        proto_emb = proto_emb.detach()
                        
                        query_emb = z_query if model.use_z_in_predict else code_query

                        # Compute logits

                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                    else:
                        pred_proto = torch.zeros(z_query.shape[0], num_classes, device=z_query.device)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)

                pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin

                # Evaluate

                value = evaluate(pred, y_query, task)
                test_values.append(value)

            return {
                'train': np.mean(train_values),
                'val': np.mean(val_values),
                'test': np.mean(test_values),
                'metric': task2metric[task]
            }
    else:
        # 【修复】添加 mini_batch 模式的评估处理
        if setting == "standard":
            # 首先需要计算原型（使用训练数据）
            if use_proto_clf:
                proto_idx = sample_proto_instances(
                    labels.cpu(),
                    mask2idx(split["train"].cpu()),
                    num_instances_per_class=num_instances_per_class,
                )
                # 如果 proto_idx 为空，使用训练数据计算原型
                if len(proto_idx) == 0:
                    # 创建一个临时的 loader 用于计算原型
                    from torch_geometric.loader import NeighborLoader
                    train_mask_indices = mask2idx(split["train"].cpu())
                    # 【显存优化】对于大数据集，减小 batch_size 和邻居采样数
                    num_nodes = dataset.x.shape[0] if hasattr(dataset, 'x') else 0
                    if num_nodes > 1000000:
                        # 大数据集（如 Reddit）：非常保守的参数
                        eval_batch_size = 128  # 进一步降低
                        eval_num_neighbors = [5, 5]  # 进一步降低邻居采样
                    else:
                        eval_batch_size = 512
                        eval_num_neighbors = kwargs.get("num_neighbors", [30] * 2)
                    
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=eval_num_neighbors,
                        input_nodes=train_mask_indices,
                        batch_size=eval_batch_size,
                        num_workers=4,  # 减少 worker 数量
                    )
                else:
                    from torch_geometric.loader import NeighborLoader
                    # 【显存优化】对于大数据集，减小 batch_size 和邻居采样数
                    num_nodes = dataset.x.shape[0] if hasattr(dataset, 'x') else 0
                    if num_nodes > 1000000:
                        eval_batch_size = 256
                        eval_num_neighbors = [10, 10]
                    else:
                        eval_batch_size = 512
                        eval_num_neighbors = kwargs.get("num_neighbors", [30] * 2)
                    
                    proto_loader = NeighborLoader(
                        dataset,
                        num_neighbors=eval_num_neighbors,
                        input_nodes=proto_idx,
                        batch_size=eval_batch_size,
                        num_workers=4,
                    )
                
                code_list, y_list = [], []
                with torch.no_grad():  # 【显存优化】评估时不需要梯度
                    for batch in proto_loader:
                        batch = batch.to(device)
                        bs = batch.batch_size
                        
                        x = batch.node_text_feat
                        edge_index = batch.edge_index
                        edge_attr = batch.edge_text_feat[batch.xe]
                        y = batch.y[:bs]
                        z = model.encode(x, edge_index, edge_attr)[:bs]
                        
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        code_list.append(code.detach().cpu())  # 立即移到 CPU 释放显存
                        y_list.append(y.cpu())
                
                if len(code_list) > 0:
                    code = torch.cat(code_list, dim=0).to(device)  # 重新移到 GPU 用于计算原型
                    y = torch.cat(y_list, dim=0).to(device)
                    
                    proto_emb = model.get_class_prototypes(code, y, num_classes)
                    if proto_emb is not None:
                        proto_emb = proto_emb.detach()
                    
                    # 释放中间变量显存
                    del code, y
                    torch.cuda.empty_cache()
                else:
                    proto_emb = None
            else:
                proto_emb = None
            
            # 使用 loader 评估所有数据
            # 由于 NeighborLoader 可能重复采样节点，我们需要去重
            # 使用字典存储每个节点的预测（使用 n_id 作为键）
            pred_dict = {}
            y_dict = {}
            mask_dict = {'train': {}, 'val': {}, 'test': {}}
            
            with torch.no_grad():  # 【显存优化】评估时不需要梯度
                for batch in loader:
                    batch = batch.to(device)
                    bs = batch.batch_size
                    
                    x = batch.node_text_feat
                    edge_index = batch.edge_index
                    edge_attr = batch.edge_text_feat[batch.xe]
                    
                    # 获取全局节点索引
                    # NeighborLoader 会在 batch.n_id 中存储原始节点索引
                    # 对于 subgraph_loader（没有指定 input_nodes），它会遍历所有节点
                    # 前 batch_size 个节点就是 input nodes
                    if hasattr(batch, 'n_id'):
                        global_node_ids = batch.n_id[:bs].cpu()
                    else:
                        # 如果没有 n_id，尝试从其他属性获取，或者使用局部索引
                        # 这种情况应该很少见
                        global_node_ids = torch.arange(bs, device='cpu')
                    
                    # 获取标签和mask
                    y_batch = labels[global_node_ids].to(device)
                    train_mask_batch = split["train"][global_node_ids].cpu()
                    val_mask_batch = split["valid"][global_node_ids].cpu()
                    test_mask_batch = split["test"][global_node_ids].cpu()
                    
                    z = model.encode(x, edge_index, edge_attr)[:bs]
                    
                    # 计算预测
                    if use_proto_clf and proto_emb is not None:
                        code, _ = model.get_codes(z, use_orig_codes=True)
                        query_emb = z if use_z_in_predict else code
                        pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)
                    else:
                        pred_proto = torch.zeros(bs, num_classes, device=device)
                    
                    if use_lin_clf:
                        pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
                    else:
                        pred_lin = torch.zeros(bs, num_classes, device=device)
                    
                    pred = (1 - model.trade_off) * pred_proto + model.trade_off * pred_lin
                    
                    # 存储每个节点的预测（使用全局节点ID作为键，避免重复）
                    for i, node_id in enumerate(global_node_ids):
                        node_id = node_id.item()
                        if node_id not in pred_dict:
                            pred_dict[node_id] = pred[i].detach().cpu()
                            y_dict[node_id] = y_batch[i].item()
                            mask_dict['train'][node_id] = train_mask_batch[i].item()
                            mask_dict['val'][node_id] = val_mask_batch[i].item()
                            mask_dict['test'][node_id] = test_mask_batch[i].item()
                    
                    # 【显存优化】每个 batch 后清理
                    del x, edge_index, edge_attr, z, pred
                    if 'code' in locals():
                        del code
                    if 'query_emb' in locals():
                        del query_emb
            
            # 转换为列表并排序（按节点ID）
            sorted_node_ids = sorted(pred_dict.keys())
            all_pred = torch.stack([pred_dict[node_id] for node_id in sorted_node_ids])
            all_y = torch.tensor([y_dict[node_id] for node_id in sorted_node_ids])
            all_train_mask = torch.tensor([mask_dict['train'][node_id] for node_id in sorted_node_ids], dtype=torch.bool)
            all_val_mask = torch.tensor([mask_dict['val'][node_id] for node_id in sorted_node_ids], dtype=torch.bool)
            all_test_mask = torch.tensor([mask_dict['test'][node_id] for node_id in sorted_node_ids], dtype=torch.bool)
            
            # 评估
            train_value = evaluate(all_pred, all_y, task, all_train_mask)
            val_value = evaluate(all_pred, all_y, task, all_val_mask)
            test_value = evaluate(all_pred, all_y, task, all_test_mask)
            
            return {
                'train': train_value,
                'val': val_value,
                'test': test_value,
                'metric': task2metric[task]
            }
        else:
            # few_shot 模式在 mini_batch 下暂时不支持，返回默认值
            raise NotImplementedError("few_shot setting with mini_batch is not implemented")