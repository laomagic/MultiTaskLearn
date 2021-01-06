import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

seed = 1999
np.random.seed(seed)
root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class GenData:
    def __init__(self, ):
        self.tnews_train = pd.read_csv(root_path + "/datasets/TNEWS/TNEWS_train1128.csv", sep="\t", quoting=3,
                                       names=['id', 'sentence', 'label'])

        self.ocnli_train = pd.read_csv(root_path + "/datasets/OCNLI/OCNLI_train1128.csv", sep="\t", quoting=3,
                                       names=['id', 'sentence1', 'sentence2', 'label'])
        self.ocnli_train['sentence'] = self.ocnli_train['sentence1'].str.cat(self.ocnli_train['sentence2'], sep="[SEP]")

        self.ocemotion_train = pd.read_csv(root_path + "/datasets/OCEMOTION/OCEMOTION_train1128.csv", sep="\t",
                                           quoting=3,
                                           names=['id', 'sentence', 'label'])

        self.tnews_test = pd.read_csv(root_path + "/datasets/TNEWS/TNEWS_a.csv", sep="\t", quoting=3,
                                      names=['id', 'sentence', 'label'])
        self.ocnli_test = pd.read_csv(root_path + "/datasets/OCNLI/OCNLI_a.csv", sep="\t", quoting=3,
                                      names=['id', 'sentence1', 'sentence2', 'label'])
        self.ocnli_test['sentence'] = self.ocnli_test['sentence1'].str.cat(self.ocnli_test['sentence2'], sep="[SEP]")
        self.ocemotion_test = pd.read_csv(root_path + "/datasets/OCEMOTION/OCEMOTION_a.csv", sep="\t", quoting=3,
                                          names=['id', 'sentence', 'label'])
        tnews_length = self.tnews_train['sentence'].map(len).values
        ocnli_length = self.ocnli_train['sentence'].map(len).values
        ocemotion_length = self.ocemotion_train['sentence'].map(len).values
        print("mean length tnews:%.2f ocnli:%.2f ocemotion:%.2f" % (sum(tnews_length)/len(tnews_length),
                                                              sum(ocnli_length)/len(ocnli_length),
                                                              sum(ocemotion_length)/len(ocemotion_length)))
        print("max length tnews:%s ocnli:%s ocemotion:%s" % (max(tnews_length),
                                                             max(ocnli_length),
                                                             max(ocemotion_length)))
        print("min length tnews:%s ocnli:%s ocemotion:%s" % (min(tnews_length),
                                                             min(ocnli_length),
                                                             min(ocemotion_length)))

        self.tnews_max_length = self.get_max_length(self.tnews_train["sentence"]) + 2
        self.ocnli_max_length = self.get_max_length(self.ocnli_train["sentence"]) + 2
        self.ocemotion_max_length = self.get_max_length(self.ocemotion_train["sentence"]) + 2

    @staticmethod
    def get_max_length(data):
        length = max(data.map(len).values)
        return length


if __name__ == '__main__':
    g = GenData()
    # 长度
    length = pd.concat([g.tnews_train['sentence'].map(len).rename('tnews'),
                        g.ocnli_train['sentence'].map(len).rename('ocnli'),
                        g.ocemotion_train['sentence'].map(len).rename('ocemotion')],
                       axis=1)
    print(length.head(5))
    length.plot.hist(bins=6)
    plt.show()
