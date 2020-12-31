from sklearn.model_selection import train_test_split
import numpy as np
from collections import Counter
import json
import math
import os

seed = 2020
np.random.seed(seed)


class Generate:
    """对原始数据进行处理"""
    def __init__(self, path, task_name):
        self.path = path
        self.task_name = task_name
        self.label_dict = {}
        self.label_weight = {}
        self.precess_data()
        pass

    def split_data(self, count):
        """分割训练集和验证集"""
        print('split :%s data' % self.task_name, "=" * 30)
        with open(os.path.join(self.path, "%s_train1128.csv" % self.task_name), 'r', encoding='utf-8') as reader:
            data = [line.strip().split("\t") for line in reader.readlines()]
        texts = [line[1] if len(line) == 3 else line[1:3] for line in data]
        labels = [line[-1] for line in data]
        test_size = round(count / len(texts), 2)
        x_train, x_dev, y_train, y_dev = train_test_split(texts, labels, test_size=test_size, shuffle=True)
        print('x_train:%s  y_train:%s  x_dev:%s  y_dev:%s' % (len(x_train), len(y_train), len(x_dev), len(y_dev)))
        label_count = Counter(y_train)
        # 标签
        self.label_dict[self.task_name] = list(label_count.keys())
        # 标签的权重
        self.label_weight[self.task_name] = [math.log(max(sum(label_count.values()), 1) / (value + 1)) for value in
                                             label_count.values()]

        def get_data(x, y):
            """迭代数据"""
            tmp_dict = {}
            for id, data in enumerate(zip(x, y)):
                if self.task_name == "OCNLI":
                    sentence1, sentence2 = data[0]
                    text = {"sentence1": sentence1, "sentence2": sentence2, "label": data[1]}
                else:
                    text = {"sentence1": data[0], "label": data[1]}
                tmp_dict[str(id)] = text
            return tmp_dict

        train = get_data(x_train, y_train)
        dev = get_data(x_dev, y_dev)
        self.write_to_json(os.path.join(self.path, "train.json"), train)  # 写入json
        self.write_to_json(os.path.join(self.path, "dev.json"), dev)
        print('split :%s data successfully' % self.task_name, "=" * 30)

    def precess_data(self):
        """处理预测数据"""
        print('precess predicting data' + "*" * 30)
        with open(os.path.join(self.path, "%s_a.csv" % self.task_name), 'r', encoding='utf-8') as reader:
            data = [line.strip().split("\t") for line in reader.readlines()]
        texts = [line[1] if len(line) == 2 else line[1:3] for line in data]
        # tmp = {"test": []}
        tmp_dict = {}
        for id, data in enumerate(texts):
            if self.task_name == "OCNLI":
                sentence1, sentence2 = data
                text = {"sentence1": sentence1, "sentence2": sentence2}
            else:
                text = {"sentence1": data}
            tmp_dict[str(id)] = text
        print("test data:%s" % len(tmp_dict))
        self.write_to_json(os.path.join(self.path, "test.json"), tmp_dict)
        print('precess predicting data successfully' + "=" * 35)

    @staticmethod
    def write_to_json(path, data):
        json.dump(data, open(path, 'w', encoding="utf-8"))


if __name__ == '__main__':
    root_path = 'datasets/'
    task_label_weight = {}
    task_label = {}
    for task_name in ["TNEWS", "OCNLI", "OCEMOTION"]:
        path = root_path + task_name
        gen = Generate(path, task_name)
        gen.split_data(count=3000)
        label_weight = gen.label_weight
        label = gen.label_dict
        task_label_weight.update(label_weight)
        task_label.update(label)
        print("*" * 50)
    print("label weight", task_label_weight)
    print("label ", task_label)
    json.dump(task_label_weight, open('./datasets/label_weight.json', 'w'))
    json.dump(task_label, open('./datasets/label.json', 'w'))
