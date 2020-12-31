import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel
from transformers.modeling_bert import BertSelfAttention, BertPooler
from transformers import AlbertModel, AlbertConfig
import transformers
from transformers import BertForSequenceClassification
from torch.nn import MSELoss, CrossEntropyLoss
import torch
import math
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "True"


class JointModel(nn.Module):
    def __init__(self, model):
        super(JointModel, self).__init__()
        self.bert = model
        self.tnews_layer = nn.Linear(768, 15)
        self.ocnli_layer = nn.Linear(768, 3)
        self.ocemotion_layer = nn.Linear(768, 7)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, task_name=None):
        hidden, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        if task_name == "TNEWS":
            return self.tnews_layer(pooled)
        elif task_name == "OCNLI":
            return self.ocnli_layer(pooled)
        elif task_name == "OCEMOTION":
            return self.ocemotion_layer(pooled)
        else:
            raise ValueError("unknown task name %s" % task_name)



