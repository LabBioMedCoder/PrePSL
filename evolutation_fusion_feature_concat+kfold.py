#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : WY Liang
# @Project  : paper_test_3
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split,KFold
from torch.utils.data import TensorDataset, DataLoader
from sklearn.svm import SVC
from sklearn.metrics import average_precision_score, label_ranking_loss, coverage_error
from sklearn.metrics import label_ranking_average_precision_score as lrap
from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold 
import matplotlib.pyplot as plt
import scipy.sparse as sp
import networkx as nx


def get_sequence_feature():
    sequence_feature_dict = {}
    # with open("./sequence_feature/sequence_feature_610_256.txt") as f_sequence:
    # with open("./sequence_feature/sequence_feature_610_512.txt") as f_sequence:
    with open("./sequence_feature/sequence_feature_610_1024.txt") as f_sequence:
    # with open("./sequence_feature/sequence_feature_610_2048.txt") as f_sequence:
        sequence_feature_content = f_sequence.readlines()
        for line in sequence_feature_content:
            protein = line.strip("\n").split(" ")[0]
            feature = line.strip("\n").split(" ")[1:]
            feature = list(map(float, feature))
            sequence_feature_dict[protein] = feature
    return sequence_feature_dict


def get_net_feature():
    net_feature_dict = {}
    # with open("./net_feature/gae_feature_256.txt") as f_net:
    # with open("./net_feature/gae_feature_512.txt") as f_net:
    # with open("./net_feature/gae_feature_1024.txt") as f_net:
    # with open("./net_feature/gae_feature_2048.txt") as f_net:
    with open("./net_feature/gcn_feature_512.txt") as f_net:
        net_feature_content = f_net.readlines()
        for line in net_feature_content:
            protein = line.strip("\n").split(" ")[0]
            feature = line.strip("\n").split(" ")[1:]
            feature = list(map(float, feature))
            net_feature_dict[protein] = feature
    return net_feature_dict


def get_label():
    subcellular_label_dict = {}
    with open("./data/protein_label_3106(0_1).txt") as f_label:
        label_content = f_label.readlines()
        for line in label_content:
            protein = line.strip("\n").split(" ")[0]
            label = line.strip("\n").split(" ")[1:]
            label = list(map(int, label))
            subcellular_label_dict[protein] = label
    return subcellular_label_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 12345
torch.manual_seed(seed_value)     # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)      # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)   # 为所有GPU设置随机种子(多块GPU)
# 定义多标签分类网络模型
class MultiLabelClassifier(nn.Module):
    def __init__(self, input_size):
        super(MultiLabelClassifier, self).__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 14)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.dropout1(out)

        out = self.fc3(out)
        out = self.sigmoid(out)

        return out


# 定义训练函数
def train_model(model, criterion, optimizer, train_loader):
    for i, (feature, labels) in enumerate(train_loader):
        feature = feature.float()
        labels = labels.float()
        # 前向传播
        outputs = model(feature)
        loss = criterion(outputs, labels)
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_mldel(epoch, ):
    result = torch.tensor([]).to(device)
    all_label = torch.tensor([]).to(device)
    with torch.no_grad():
        for i, (feature, labels) in enumerate(test_loader):
            feature = feature.float()
            labels = labels.float()
            outputs = model(feature)
            result = torch.cat((result, outputs))
            all_label = torch.cat((all_label, labels))

    result = result.cpu()
    all_label = all_label.cpu()
    r = label_ranking_loss(all_label, result)
    c = coverage_error(all_label, result) - 1
    a = lrap(all_label, result)

    global min_ranking_loss
    global min_coverage_error
    global max_average_precision_score
    global best_epoch
    global single_result
    if a > max_average_precision_score:
        min_ranking_loss = r
        min_coverage_error = c
        max_average_precision_score = a
        best_epoch = epoch
        single_result = result
    # if (epoch+1)%100 == 0:
    #     print("训练轮数为：", epoch + 1, "排名损失：", r, "覆盖误差：", c, "平均精度损失：", a)

if __name__ == '__main__':
    sequence_feature_dict = get_sequence_feature()
    net_feature_dict = get_net_feature()
    label_dict = get_label()
    print("特征顺序与标签顺序是否相同：",list(sequence_feature_dict.keys()) == list(net_feature_dict.keys()) == list(label_dict.keys()))
    
    all_label = list(label_dict.values())

    sequence_feature = list(sequence_feature_dict.values())  # 序列特征
    net_feature = list(net_feature_dict.values())  # 网络特征
    concat_feature = []  # 拼接特征
    for i in range(len(sequence_feature)):
        f = list(sequence_feature[i]) + list(net_feature[i])
        concat_feature.append(f)
    
    # print("使用序列特征！！")
    # all_feature = torch.tensor(sequence_feature).to(device)
    print("使用网络特征！！")
    all_feature = torch.tensor(net_feature).to(device)
    # print("使用融合特征！！")
    # all_feature = torch.tensor(concat_feature).to(device)
    
    print("特征大小：", all_feature.shape)
    all_label = torch.tensor(all_label).to(device)
    print("标签大小：", all_label.shape)    
  
    average_ranking_loss = 0
    average_coverage_error = 0
    average_average_precision = 0
    all_reault = {}
    all_result_oredr = {}
    
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=10)
    for k,(train_index,test_index) in enumerate(kfold.split(all_feature, all_label)):
        single_result = torch.tensor([]).to(device)
        min_ranking_loss = 0
        min_coverage_error = 0
        max_average_precision_score = 0
        best_epoch = 0
        
        X_train = all_feature[train_index]
        X_test = all_feature[test_index]
        y_train = all_label[train_index]
        y_test = all_label[test_index]
    
        train_data = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_data, shuffle=False, batch_size=64)
        test_data = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=64)



        # 定义数据集和数据加载器
        input_size = all_feature.shape[1]  # concat特征维度
        num_epochs = 1750  # 训练轮数
        learning_rate = 0.0001  # 学习率

        # 创建多标签分类模型
        model = MultiLabelClassifier(input_size)
        model = model.to(device)
        # print(next(model.parameters()).device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # 训练模型
        for epoch in range(num_epochs):
            train_model(model, criterion, optimizer, train_loader)
            test_mldel(epoch)
        print("第{}折的结果为：".format(k+1))
        print(best_epoch,"RL",min_ranking_loss, "CV",min_coverage_error, "AP",max_average_precision_score)
        for index,result in zip(test_index.tolist(), single_result.numpy().tolist()):
            all_reault[str(index)] = result
        average_ranking_loss += min_ranking_loss
        average_coverage_error += min_coverage_error
        average_average_precision += max_average_precision_score
        # break
    print("平均结果为：",average_ranking_loss/5, average_coverage_error/5, average_average_precision/5)
    
    for i in range(len(all_reault)):
        key = str(i)
        value = all_reault[key]
        all_result_oredr[key] = value
    # print(list(all_result_oredr.keys()))
    print(np.array(list(all_result_oredr.values())).shape)
    
    with open("all_result_order.txt","a") as f_result_oredr:
        for k,v in all_result_oredr.items():
            v = list(map(str, v))
            f_result_oredr.write(k+" "+" ".join(v)+"\n")
    
    r = label_ranking_loss(all_label.cpu(), np.array(list(all_result_oredr.values())))
    c = coverage_error(all_label.cpu(), np.array(list(all_result_oredr.values()))) - 1
    a = lrap(all_label.cpu(), np.array(list(all_result_oredr.values())))
    print(r,c,a)
