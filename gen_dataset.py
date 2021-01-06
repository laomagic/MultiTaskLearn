import os
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, SubsetRandomSampler
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from itertools import zip_longest
from tqdm import tqdm
import math
from collections import Counter
import json
import numpy as np
import csv


seed = 1999
np.random.seed(seed)
root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def convert_features(data, labels, tokenizer, max_length):
    input_ids = []
    attention_mask = []
    token_type_ids = []
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        sentence = row["sentence"]
        text_dict = tokenizer.encode_plus(sentence, pad_to_max_length=True, max_length=max_length, truncation=True,
                                          return_attention_mask=True, return_tensors="pt")
        input_ids.append(text_dict["input_ids"])
        attention_mask.append(text_dict["attention_mask"])
        token_type_ids.append(text_dict["token_type_ids"])
    input_ids = torch.cat(input_ids, dim=0)
    attention_mask = torch.cat(attention_mask, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)
    if labels is None:
        return TensorDataset(input_ids, attention_mask, token_type_ids)
    labels = torch.tensor(labels.values)
    return TensorDataset(input_ids, attention_mask, token_type_ids, labels)


class GenDatasets:
    def __init__(self, tokenizer, batch_size=1, sampler=True):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.label_dict = {}
        self.label_weight = {}
        tnews_train = pd.read_csv(root_path + "/datasets/TNEWS/TNEWS_train1128.csv", sep="\t",
                                  quoting=csv.QUOTE_NONE,
                                  names=['id', 'sentence', 'label'])

        ocnli_train = pd.read_csv(root_path + "/datasets/OCNLI/OCNLI_train1128.csv", sep="\t",
                                  quoting=csv.QUOTE_NONE,
                                  names=['id', 'sentence1', 'sentence2', 'label'])
        ocemotion_train = pd.read_csv(root_path + "/datasets/OCEMOTION/OCEMOTION_train1128.csv", sep="\t",
                                      quoting=csv.QUOTE_NONE,
                                      names=['id', 'sentence', 'label'])

        ocnli_train['sentence'] = ocnli_train['sentence1'].str.cat(ocnli_train['sentence2'], sep="[SEP]")

        tnews_test = pd.read_csv(root_path + "/datasets/TNEWS/tnews_test_B.csv", sep="\t",
                                 quoting=csv.QUOTE_NONE,
                                 names=['id', 'sentence', 'label'])
        ocnli_test = pd.read_csv(root_path + "/datasets/OCNLI/ocnli_test_B.csv", sep="\t",
                                 quoting=csv.QUOTE_NONE,
                                 names=['id', 'sentence1', 'sentence2', 'label'])
        ocnli_test['sentence'] = ocnli_test['sentence1'].str.cat(ocnli_test['sentence2'], sep="[SEP]")
        ocemotion_test = pd.read_csv(root_path + "/datasets/OCEMOTION/ocemotion_test_B.csv", sep="\t",
                                     quoting=csv.QUOTE_NONE,
                                     names=['id', 'sentence', 'label'])
        # self.max_length = max(self.get_max_length(tnews_train["sentence"]),
        #                       self.get_max_length(ocnli_train["sentence"]),
        #                       self.get_max_length(ocemotion_train["sentence"])) + 2
        self.tnews_max_length = 128
        # self.tnews_max_length = self.get_max_length(tnews_train["sentence"]) + 2
        # self.ocnli_max_length = self.get_max_length(ocnli_train["sentence"]) + 2
        self.ocnli_max_length = 128
        self.ocemotion_max_length = 128
        # self.ocemotion_max_length = self.get_max_length(ocemotion_train["sentence"]) + 2


        self.generate_weight(tnews_train["label"].values, "TNEWS")
        self.generate_weight(ocnli_train["label"].values, "OCNLI")
        self.generate_weight(ocemotion_train["label"].values, "OCEMOTION")
        json.dump(self.label_weight, open("label_weights.json", "w", encoding="utf-8"))
        tnews_length = len(tnews_train)
        ocnli_length = len(ocnli_train)
        ocemotion_length = len(ocemotion_train)
        length = tnews_length + ocnli_length + ocemotion_length
        tnews_ratio = int((tnews_length * 0.9 / length) * self.batch_size)
        ocnli_ratio = int((ocnli_length * 0.9 / length) * self.batch_size)
        # ocemotion_ratio = int((ocemotion_length * 0.9 / length) * self.batch_size)
        ocemotion_ratio = self.batch_size - tnews_ratio - ocnli_ratio

        tnews_train_dataset, tnews_dev_dataset, tnews_test_dataset, self.tnews_id2label = \
            self.get_dataset(tnews_train, tnews_test, self.tnews_max_length)
        ocnli_train_dataset, ocnli_dev_dataset, ocnli_test_dataset, self.ocnli_id2label = \
            self.get_dataset(ocnli_train, ocnli_test, self.ocnli_max_length)
        ocemotion_train_dataset, ocemotion_dev_dataset, ocemotion_test_dataset, self.ocemotion_id2label = \
            self.get_dataset(ocemotion_train, ocemotion_test, self.ocemotion_max_length)

        len_tnews = len(tnews_train_dataset)  # 57024
        len_ocnli = len(ocnli_train_dataset)  # 43900
        len_ocemotion = len(ocemotion_train_dataset)  # 31783
        len_max = max(len_tnews, len_ocnli, len_ocemotion)
        if sampler:
            self.tnews_train_dataloader = DataLoader(tnews_train_dataset,
                                                     sampler=RandomSampler(tnews_train_dataset),
                                                     batch_size=tnews_ratio)
            self.ocnli_train_dataloader = DataLoader(ocnli_train_dataset,
                                                     sampler=RandomSampler(ocnli_train_dataset,
                                                                           replacement=True,
                                                                           num_samples=len_max),
                                                     batch_size=ocnli_ratio)
            self.ocemotion_train_dataloader = DataLoader(ocemotion_train_dataset,
                                                         sampler=RandomSampler(ocemotion_train_dataset,
                                                                               replacement=True,
                                                                               num_samples=len_max),
                                                         batch_size=ocemotion_ratio)
        else:
            self.tnews_train_dataloader = DataLoader(tnews_train_dataset,
                                                     sampler=RandomSampler(tnews_train_dataset),
                                                     batch_size=tnews_ratio)
            self.ocnli_train_dataloader = DataLoader(ocnli_train_dataset,
                                                     sampler=RandomSampler(ocnli_train_dataset),
                                                     batch_size=ocnli_ratio)
            self.ocemotion_train_dataloader = DataLoader(ocemotion_train_dataset,
                                                         sampler=RandomSampler(ocemotion_train_dataset),
                                                         batch_size=ocemotion_ratio)
        self.tnews_dev_dataloader, self.ocnli_dev_dataloader, self.ocemotion_dev_dataloader = \
            self.gen_dataloaders(tnews_dev_dataset, ocnli_dev_dataset, ocemotion_dev_dataset)
        self.tnews_test_dataloader, self.ocnli_test_dataloader, self.ocemotion_test_dataloader = \
            self.gen_dataloaders(tnews_test_dataset, ocnli_test_dataset, ocemotion_test_dataset)

    def generate_weight(self, data, task_name):
        label_count = Counter(data)
        self.label_dict[task_name] = set(data)
        self.label_weight[task_name] = [math.log(max(sum(label_count.values()), 1) / (value + 1)) for value in
                                        label_count.values()]

    @staticmethod
    def get_max_length(data):
        length = max(data.map(len).values)
        return length

    def get_dataset(self, train, test, test_ratio=0.1, max_length=128):
        labels = set(train['label'].values)
        label2id = {}
        id2label = {}
        id = 0
        for label in labels:
            label2id[label] = id
            id += 1
        for label, id in label2id.items():
            id2label[id] = label

        train['label'] = train['label'].map(label2id)
        x_train, x_dev, y_train, y_dev = train_test_split(train[["sentence"]],
                                                          train['label'],
                                                          test_size=test_ratio,
                                                          shuffle=True,
                                                          random_state=2020)

        train_dataset = convert_features(x_train, y_train, self.tokenizer, max_length)
        dev_dataset = convert_features(x_dev, y_dev, self.tokenizer, max_length)
        test_dataset = convert_features(test[["sentence"]], None, self.tokenizer, max_length)

        return train_dataset, dev_dataset, test_dataset, id2label

    def gen_dataloaders(self, *dataset):
        datasets = []
        for d in dataset:
            # datasets.append(DataLoader(d, sampler=SequentialSampler(d), batch_size=self.batch_size))
            datasets.append(DataLoader(d, batch_size=self.batch_size))
        return datasets


if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(root_path + "/model/w/")

    g = GenDatasets(tokenizer, batch_size=64, sampler=False)
    for batch in g.tnews_dev_dataloader:
        print(batch[0])
