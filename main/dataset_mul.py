import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
from bert_config import Config
from utils import clean_symbols, rm_stop_word
import jieba
import os
seed = 9999
np.random.seed(seed)
os.environ["TOKENIZERS_PARALLELISM"] = "True"


class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, bert_tokenizer, path, max_length=128):
        """
        bert_tokenizer :分词器
        LCQMC_file     :语料文件
        """
        super(DataPrecessForSentence, self).__init__()
        self.bert_tokenizer = bert_tokenizer
        self.max_length = max_length
        self.seqs, self.seq_masks, self.seq_segments, self.labels \
            = self.get_input(path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.seq_masks[idx], self.seq_segments[
            idx], self.labels[idx]

    # 获取文本与标签
    def get_input(self, path):
        """
        通对输入文本进行分词、ID化、截断、填充等流程得到最终的可用于模型输入的序列。
        入参:
            dataset     : pandas的dataframe格式，包含三列，第一,二列为文本，第三列为标签。标签取值为{0,1}，其中0表示负样本，1代表正样本。
            max_seq_len : 目标序列长度，该值需要预先对文本长度进行分别得到，可以设置为小于等于512（BERT的最长文本序列长度为512）的整数。
        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          那么取值为1，否则为0。
            seq_segment : shape等于seq，因为是单句，所以取值都为0。
            labels      : 标签取值为{0,1}，其中0表示负样本，1代表正样本。
        """

        #                 )
        # path = json.load(open('config/task.json', 'r'))['task_name'][task_name][mode]

        with open(path, 'r', encoding='utf-8') as reader:
            data = [line.strip().split('\t') for line in reader.readlines()]
        #         df = df[df['question1'].isin(
        #             np.random.choice(df['question1'].unique(),
        #                              int(0.3 * df['question1'].unique().shape[0])))]
        # 去除特殊符号
        text = [clean_symbols(line[1]) for line in data]
        labels = [int(line[2]) for line in data]

        # 切词
        tokens_seq = tqdm(list(map(jieba.lcut, text)))  # jieba分词版本
        # tokens_seq = tqdm(list(map(self.bert_tokenizer.tokenize, text)))  # char 字符
        # 去除停用词
        tokens_seq = tqdm(list(map(rm_stop_word, tokens_seq)))

        # 获取定长序列及其mask
        result = tqdm(list(map(self.trunate_and_pad, tokens_seq)))
        seqs = [i[0] for i in result]
        seq_masks = [i[1] for i in result]
        seq_segments = [i[2] for i in result]
        return torch.tensor(seqs).type(torch.long), torch.tensor(seq_masks).type(torch.long), torch.tensor(
            seq_segments).type(torch.long), torch.tensor(labels).type(torch.long)

    def trunate_and_pad(self, tokens_seq):
        """
            token_seq       : 输入序列，在本处其为单个句子。
            max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度

        出参:
            seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
            seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
                          取值为0，否则为1。
            seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分

        """
        # 对超长序列进行截断
        if len(tokens_seq) > (self.max_length - 2):
            tokens_seq = tokens_seq[0:(self.max_length - 2)]

        # 句子拼接
        seq = ['[CLS]'] + tokens_seq + ['[SEP]']
        seq_segment = [0] * (len(tokens_seq) + 2)

        # 转化为id
        seq = self.bert_tokenizer.convert_tokens_to_ids(seq)
        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_length - len(seq))
        # 创建seq_mask
        seq_mask = [1] * len(seq) + padding
        # 创建seq_segment
        seq_segment = seq_segment + padding
        # 对seq拼接填充序列
        seq += padding
        assert len(seq) == self.max_length
        assert len(seq_mask) == self.max_length
        assert len(seq_segment) == self.max_length
        return seq, seq_mask, seq_segment


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """

    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)  # # tnews 索引为0 ocnli索引为1 emotion索引为2
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]  # tnews 索引为0 ocnli索引为1 emotion索引为2
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)

        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]  # 根据索引 获取task name
            yield next(dataloader_iter_dict[task_name]), task_name


if __name__ == '__main__':
    config = Config()
    # model_name = 'model/wobert/'
    # model = MultitaskModel.create(
    #     model_name=model_name,
    #     model_type_dict={
    #         "tnews": transformers.AutoModelForSequenceClassification,
    #         "ocnli": transformers.AutoModelForSequenceClassification,
    #         "ocemotion": transformers.AutoModelForSequenceClassification,
    #     },
    #     model_config_dict={
    #         "tnews": transformers.AutoConfig.from_pretrained(model_name, num_labels=15),
    #         "ocnli": transformers.AutoConfig.from_pretrained(model_name, num_labels=3),
    #         "ocemotion": transformers.AutoConfig.from_pretrained(model_name, num_labels=7),
    #     },
    # )
    tokenizer = BertTokenizer.from_pretrained('model/w/')
    train_dataset = {"tnews": {},
                     "ocnli": {},
                     "ocemotion": {}}
    tnews_train_dataset = DataPrecessForSentence(tokenizer, config.tnews_train_path)
    ocnli_train_dataset = DataPrecessForSentence(tokenizer, config.ocnli_train_path)
    emotion_train_dataset = DataPrecessForSentence(tokenizer, config.emotion_train_path)
    train_dataset['tnews']["train"] = tnews_train_dataset
    train_dataset['ocnli']["train"] = ocnli_train_dataset
    train_dataset['ocemotion']["train"] = emotion_train_dataset
    print(tnews_train_dataset.__len__())
    print(ocnli_train_dataset.__len__())
    print(emotion_train_dataset.__len__())
    dataloader = {}
    tnews_dataloader = DataLoader(train_dataset['tnews']['train'],
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
    dataloader['tnews'] = tnews_dataloader
    ocnli_dataloader = DataLoader(train_dataset['ocnli']['train'],
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
    dataloader['ocnli'] = ocnli_dataloader
    emotion_dataloader = DataLoader(train_dataset['ocemotion']['train'],
                                    batch_size=64,
                                    shuffle=True,
                                    num_workers=2,
                                    drop_last=True)
    dataloader['ocemotion'] = emotion_dataloader
    # bconfig = BertConfig()
    # model_name = 'model/wobert'

    # model = Multitask.create(
    #     model_name=model_name,
    #     model_type_dict={
    #         "tnews": BertBaseModel(bconfig),
    #         "ocnli": BertBaseModel(bconfig),
    #         "ocemotion": BertBaseModel(bconfig),
    #     },
    #     model_config_dict={
    #         "tnews": transformers.AutoConfig.from_pretrained(model_name, num_labels=15),
    #         "ocnli": transformers.AutoConfig.from_pretrained(model_name, num_labels=3),
    #         "ocemotion": transformers.AutoConfig.from_pretrained(model_name, num_labels=7),
    #     },
    # )
    multi = MultitaskDataloader(dataloader)
    # print(next(iter(multi)))
    total = 0
    for step,(batch,task_name) in enumerate(multi):
        total +=1
    print(total)