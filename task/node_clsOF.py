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
        #OFA要做的
        proto_loss = torch.tensor(0.0, device=device)
        act_loss = torch.tensor(0.0, device=device)

        if use_proto_clf:
            # Compute Prototypes
            code_train, commit_loss = model.get_codes(z_train, use_orig_codes=True)

            proto_emb = model.get_class_prototypes(code_train, y_train, num_classes).detach()
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
            #OFA要做的
            proto_loss = torch.tensor(0.0, device=device)
            act_loss = torch.tensor(0.0, device=device)
            
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

                proto_emb = model.get_class_prototypes(code_train, y_train, num_classes).detach()
                query_emb = z if model.use_z_in_predict else code

                pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

            if use_lin_clf:
                pred_lin = model.get_lin_logits(z).mean(1).softmax(dim=-1)
                # Initialize pred_proto with correct shape if not set
                #OFA要做的
                if not use_proto_clf:
                    pred_proto = torch.zeros_like(pred_lin)

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

                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes).detach()
                    query_emb = z_query if use_z_in_predict else code_query

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)
                    # Initialize pred_proto with correct shape if not set
                    #OFA要做的
                    if not use_proto_clf:
                        pred_proto = torch.zeros_like(pred_lin)

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

                    proto_emb = model.get_class_prototypes(code_support, y_support, num_classes).detach()

                    query_emb = z_query if model.use_z_in_predict else code_query

                    # Compute logits

                    pred_proto = model.get_proto_logits(query_emb, proto_emb).softmax(dim=-1)

                if use_lin_clf:
                    pred_lin = model.get_lin_logits(z_query).mean(1).softmax(dim=-1)
                    # Initialize pred_proto with correct shape if not set
                    #OFA要做的
                    if not use_proto_clf:
                        pred_proto = torch.zeros_like(pred_lin)

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