import importlib
import os
from ofa.data.ofa_data import OFAPygDataset

# 【修改点】在AVAILABLE_DATA列表中添加Citeseer、Photo、Computers、Reddit，使其可被SingleGraphOFADataset识别
AVAILABLE_DATA = ["Cora", "Pubmed", "Citeseer", "Photo", "Computers", "Reddit", "wikics", "arxiv"]


class SingleGraphOFADataset(OFAPygDataset):
    def gen_data(self):
        if self.name not in AVAILABLE_DATA:
            raise NotImplementedError("Data " + self.name + " is not implemented")
        data_module = importlib.import_module("ofa.data.single_graph." + self.name + ".gen_data")
        return data_module.get_data(self.data_dir)

    def add_text_emb(self, data_list, text_emb):
        data_list[0].node_text_feat = text_emb[0]
        data_list[0].edge_text_feat = text_emb[1]
        # data_list[0].noi_node_text_feat = text_emb[2]
        data_list[0].class_node_text_feat = text_emb[3]
        # data_list[0].prompt_edge_text_feat = text_emb[4]
        return self.collate(data_list)

    def get_task_map(self):
        return self.side_data

    def get_edge_list(self, mode="e2e"):
        if mode == "e2e_node":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}
        elif mode == "lr_node":
            return {"f2n": [1, 0]}
        elif mode == "e2e_link":
            return {"f2n": [1, 0], "n2f": [3, 0], "n2c": [2, 0], "c2n": [4, 0]}