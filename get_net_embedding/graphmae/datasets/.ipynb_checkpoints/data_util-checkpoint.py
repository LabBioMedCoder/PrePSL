
from collections import namedtuple, Counter
import numpy as np

import torch
import torch.nn.functional as F

import dgl
from sklearn.decomposition import PCA

def load_dataset(dataset):
    print(dataset)
    if dataset == "data":
        graph = dgl.DGLGraph()
        graph.add_nodes(3106)
        head_node = []
        tail_node = []
        with open("../data/edge.txt") as f_edge:
            edge_content = f_edge.readlines()
            for line in edge_content:
                head = int(line.strip("\n").split(" ")[0])
                head_node.append(head)
                tail = int(line.strip("\n").split(" ")[1])
                tail_node.append(tail)
        graph.add_edges(head_node, tail_node)

        graph = dgl.add_self_loop(graph)

        print("图中节点数量为：", graph.num_nodes())
        print("图中边的数量为：", graph.num_edges())

        all_protein = []
        features = []
        with open("../data/protein_feature_3106(0_1).txt") as f_feature:
            feature_content = f_feature.readlines()
            for line2 in feature_content:
                protein = line2.strip("\n").split(" ")[0]
                f = line2.strip("\n").split(" ")[1:]
                f = list(map(int, f))
                features.append(f)
                all_protein.append(protein)
        graph.ndata["feat"] = torch.Tensor(features)
        print("图中特征数量为：", graph.ndata["feat"].shape)

        all_protein_2 = []
        labels = []
        with open("../data/protein_label_3106(0_1).txt") as f_label:
            label_content = f_label.readlines()
            for line3 in label_content:
                p = line3.strip("\n").split(" ")[0]
                all_protein_2.append(p)
                l = line3.strip("\n").split(" ")[1:]
                l = list(map(int, l))
                labels.append(l)

        print("特征顺序与标签顺序是否相同：", all_protein==all_protein_2)

        graph.ndata["label"] = torch.Tensor(labels).long()

        num_features = graph.ndata["feat"].shape[1]


        return graph, num_features, all_protein