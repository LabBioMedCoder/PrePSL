import logging
import numpy as np
from tqdm import tqdm
import torch

from graphmae.utils import build_args,create_optimizer,set_random_seed,load_best_configs
from graphmae.datasets.data_util import load_dataset
from graphmae.models import build_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graph, feat, optimizer, max_epoch, device):
    logging.info("start training..")
    graph = graph.to(device)
    x = feat.to(device)

    epoch_iter = tqdm(range(max_epoch))

    for epoch in epoch_iter:
        model.train()
        loss, loss_dict = model(graph, x)       # 传到forward里面进行训练
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")
    return model


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = args.seeds
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    optim_type = args.optimizer
    lr = args.lr
    weight_decay = args.weight_decay

    graph, num_features, all_protein = load_dataset(dataset=dataset_name)
    print(type(graph), type(num_features), type(all_protein))
    args.num_features = num_features


    print("使用图自编码器")
    set_random_seed(seeds)
    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    x = graph.ndata["feat"].to(device)
    print("原始节点特征维度为：",x.shape)
    print("开始预训练")
    model = pretrain(model, graph, x, optimizer, max_epoch, device)


    all_feature = model.embed(graph.to(device), x.to(device))
    print("使用图自编码器得到的特征维度：", all_feature.shape)
    all_feature = all_feature.cpu().detach().numpy().tolist()
    # all_label = graph.ndata["label"].cpu().numpy().tolist()

    with open("../net_feature/gae_feature.txt", "a") as f_feature:
        for i in range(len(all_protein)):
            feature = list(map(str,all_feature[i]))
            f_feature.writelines(all_protein[i]+" "+" ".join(feature)+"\n")

if __name__ == "__main__":
    args = build_args()
    args = load_best_configs(args, "configs.yml")
    main(args)
