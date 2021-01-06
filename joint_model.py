import torch.nn as nn
from transformers import BertModel, BertConfig, BertPreTrainedModel, RobertaConfig, RobertaModel, PretrainedConfig, \
    BertTokenizer
from transformers.modeling_bert import BertSelfAttention, BertPooler
from transformers import AlbertModel, AlbertConfig
from bert_config import Config
import transformers
from transformers import BertForSequenceClassification
from torch.nn import MSELoss, CrossEntropyLoss
import torch
import math
import json


class BertJointModel(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(BertJointModel, self).__init__(bert_config)
        # self.bert = BertModel
        self.bert = BertModel(bert_config)
        # self.bert = RobertaModel.from_pretrained('model/rbtl3_private/')
        # self.bert = AutoModel.from_pretrained(bert_config.model_name)
        self.num_labels = bert_config.num_labels
        self.dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        self.fc = nn.Linear(2 * bert_config.hidden_size, bert_config.hidden_size)
        self.cls = nn.Linear(bert_config.hidden_size, self.num_labels)
        self.embedding = nn.Embedding(bert_config.vocab_size, bert_config.hidden_size, padding_idx=1)
        # self.lstm = nn.LSTM(input_size=config.hidden_size,
        #                     hidden_size=config.hidden_size,
        #                     num_layers=2,
        #                     bidirectional=True,
        #                     batch_first=True)
        self.gru = nn.GRU(input_size=bert_config.hidden_size,
                          hidden_size=bert_config.hidden_size,
                          num_layers=2,
                          batch_first=True)



    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None,task_name=None):
        hidden_states, pooled = self.bert(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)  # hidden_size [batch_size,max_length,hidden_size]
        embed = self.embedding(input_ids)
        out, _ = self.gru(embed)
        # dropout
        out = self.dropout(out)
        hidden_states = self.dropout(hidden_states)
        hidden = torch.cat([out[:, -1, :], hidden_states[:, -1, :]], dim=-1)  # 两者最后一层拼接

        hidden = self.fc(hidden)
        logits = self.cls(hidden)
        return logits
        # outputs = (logits,)
        # if labels is not None:
        #     if self.num_labels == 1:
        #         #  We are doing regression
        #         loss_fct = MSELoss()
        #         loss = loss_fct(logits.view(-1), labels.view(-1))
        #     else:
        #         loss_fct = CrossEntropyLoss(self.weights[task_name.upper()])
        #         loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        #         # torch.nn.L2
        #     outputs = (loss,) + outputs
        # return outputs  # (loss), logits


class BertRNNModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertRNNModel, self).__init__(config)
        self.bert = BertModel(config)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.cls = nn.Linear(config.hidden_size, config.num_labels)

        self.lstm = nn.LSTM(input_size=config.hidden_size,
                            hidden_size=config.hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True)
        # self.gru = nn.GRU(input_size=opconfig.hidden_size,
        #                   hidden_size=opconfig.hidden_size,
        #                   num_layers=opconfig.num_layers,
        #                   bidirectional=opconfig.bidiractional,
        #                   batch_first=True)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        hidden_states, pooled = self.bert(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)  # hidden_size [batch_size,max_length,hidden_size]
        hidden_states, _ = self.lstm(hidden_states)  # hidden_size
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states[:, -1, :]  # 取最后一层
        # hidden_states = hidden_states.mean(dim=1)  # 所有token的平均值
        hidden_states = self.fc(hidden_states)
        logits = self.cls(hidden_states)

        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits


class BertBaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super(BertBaseModel, self).__init__(config)
        self.bert = BertModel(config)
        # self.bert = AlbertModel.from_pretrained(config.albert_path)
        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        # self.attention = BertSelfAttention(config)
        # self.attention = BERTSelfAttention(config)
        # self.ocnli_fc = nn.Linear(config.hidden_size, config.ocnli_classes)
        # self.emotion_fc = nn.Linear(config.hidden_size, config.emotion_classes)
        # self.fc['tnews'] = self.tnews_fc
        # self.fc['ocnli'] = self.ocnli_fc
        # self.fc['emotion'] = self.emotion_fc
        self.selfattetnion = BertSelfAttention(config)
        self.pooled = BertPooler(config)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # hidden_states [batch_size,max_length,hidden_states]
        hidden_states, pooled = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                          token_type_ids=token_type_ids)
        # outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pooled = outputs[1]
        # pooled_output = self.dropout(pooled)
        # print('pooled', pooled_output.shape)
        # attention = self.attention(pooled_output, None, use_attention_mask=False)
        hidden_states = self.selfattetnion(hidden_states)
        hidden_states = hidden_states[0].mean(dim=1)  # 所有token的平均值
        pooled = self.dense(hidden_states)
        pooled = self.activation(pooled)
        # pooled = self.pooled(hidden_states)
        pooled = self.dropout(pooled)
        # pooled_output = self.fc(attention)
        logits = self.classifier(pooled)
        outputs = (logits,)
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), logits, (hidden_states), (attentions)


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):

        return self.taskmodels_dict[task_name](**kwargs)


class Multitask(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name])

            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):

        return self.taskmodels_dict[task_name](**kwargs)


if __name__ == '__main__':
    config = Config()
    bert_config = BertConfig()
    # bert_config = RobertaConfig()
    # bert_config = RobertaConfig.from_json_file(config.config_path)
    # print(bert_config)

    # bert = BertModel.from_pretrained(config.bert_path)
    # print(bert)
    # print(bert.__class__.__name__)
    # shared_encoder = getattr(bert, 'bert')
    # print(shared_encoder)
    # model_name = 'model/rbtl3_private/'
    model_name = 'roberta-base'
    multitask_model = Multitask.create(
        model_name=model_name,
        model_type_dict={
            "tnews": BertJointModel(bert_config),
            "ocnli": BertJointModel(bert_config),
            "ocemotion": BertJointModel(bert_config),
        },
        model_config_dict={
            "tnews": transformers.AutoConfig.from_pretrained(model_name, num_labels=config.news_classes),
            "ocnli": transformers.AutoConfig.from_pretrained(model_name, num_labels=config.ocnli_classes),
            "ocemotion": transformers.AutoConfig.from_pretrained(model_name, num_labels=config.emotion_classes),
        },
    )
    print(multitask_model)
