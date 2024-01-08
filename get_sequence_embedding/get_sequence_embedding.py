#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : WY Liang
# @Project  : paper_test_3

from transformers import BertModel, BertTokenizer, BertConfig
import re
from tqdm import tqdm
import os


sequence_path = "../data/protein_sequence_3106.txt"

def get_all_sequence():
    sequence_average_length = 0
    all_protein_sequence_dict = {}
    with open(sequence_path) as f_sequence:
        sequence_content = f_sequence.read().split(">")[1:]
        for line in sequence_content:
            protein = line.strip("\n").split("\n")[0]
            sequence = line.strip("\n").split("\n")[1]
            sequence_average_length += len(sequence)
            all_protein_sequence_dict[protein] = sequence
    sequence_average_length = sequence_average_length / 3106
    return all_protein_sequence_dict, sequence_average_length


BERT_PATH = '../PLM'
def get_sequence_feature(length=610):
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH, ignore_mismatched_sizes=True)
    model_config = BertConfig.from_pretrained(BERT_PATH)
    D = str(model_config).split("\n")[8].split(":")[1][:-1].strip()
    print("序列特征维度为：",D)
    model = BertModel.from_pretrained(BERT_PATH, ignore_mismatched_sizes=True)

    print("开始获取序列特征")
    with open("../sequence_feature/sequence_feature_"+str(length)+"_"+D+".txt", "a") as f_feature:
        for protein, sequence in tqdm(all_protein_sequence_dict.items()):
            if len(sequence) > length:
                sequence = sequence[:int(length/2)]+sequence[-(int(length/2)):]
            sequence_Example = " ".join(sequence)
            sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
            encoded_input = tokenizer.encode_plus(sequence_Example, return_tensors='pt')
            output = model(**encoded_input)
            feature = output.last_hidden_state[0].mean(0).detach().numpy().tolist()
            feature = list(map(str, feature))
            f_feature.writelines(protein+" "+" ".join(feature)+"\n")

if __name__ == '__main__':


    all_protein_sequence_dict, sequence_average_length = get_all_sequence()
    print(len(all_protein_sequence_dict), sequence_average_length)
    get_sequence_feature()

