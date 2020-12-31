import os
import json
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
import itertools
from tqdm import tqdm

root_path = os.path.abspath(os.path.dirname(__file__))


class JDataset(Dataset):
    def __init__(self, tokenizer, task_name, mode):
        # self.ocnli = self.read_json("OCNLI")
        # self.ocemotion = self.read_json("OCEMOTION")
        # self.batch_size = batch_size
        path = root_path + "/datasets/" + task_name + "/" + mode + ".json"
        self.label = json.load(open(root_path + "/datasets/label.json"))[task_name]
        self.label2id = dict(zip(self.label, list(range(len(self.label)))))
        self.data = list(json.load(open(path, 'r', encoding="utf-8")).values())
        # max_length = self.get_max_length(self.data)
        # max_length = self.get_max_length(self.data)
        # print(max_length)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        line = self.data[index]
        label = self.label2id.get(line.get("label", None), 0)
        sentence1 = line.get("sentence1")
        sentence = self.tokenizer.encode_plus(sentence1,
                                               add_special_tokens=True,
                                               return_attention_mask=True)
        sentence2 = line.get("sentence2", None)
        if sentence2 is not None:
            sentence = self.tokenizer.encode_plus(sentence1,
                                                  sentence2,
                                                  add_special_tokens=True,
                                                  return_attention_mask=True)
        input_ids, attention_mask, token_type_ids = sentence['input_ids'], \
                                                    sentence['attention_mask'], \
                                                    sentence['token_type_ids']
        return {"input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "label": label}


def collate_fn(batch):
    """动态padding,返回Tensor"""
    def padding(indice, max_length, pad_idx=0):
        """
        填充每个batch的句子长度
        """
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice, dtype=torch.long)

    input_ids = [data["input_ids"] for data in batch]
    attention_mask = [data["attention_mask"] for data in batch]
    token_type_ids = [data["token_type_ids"] for data in batch]
    max_length = max([len(t) for t in input_ids])  # batch中样本的最大的长度
    labels = torch.tensor([data["label"] for data in batch], dtype=torch.long)

    input_ids_padded = padding(input_ids, max_length)  # 填充每个batch的sample
    attention_mask_padded = padding(attention_mask, max_length)  # 填充每个batch的sample
    token_type_ids_padded = padding(token_type_ids, max_length)  # 填充每个batch的sample
    return input_ids_padded, attention_mask_padded, token_type_ids_padded, labels