from transformers import BertTokenizer, BertModel
# from generator import JointDataset, JDataset,collate_fn
from bert_config import Config
from dataset_mul import DataPrecessForSentence, MultitaskDataloader
from torch.utils.data import DataLoader
import torch
# from multi_model import JointBertModel
from bert_model import JointModel
from sklearn.metrics import f1_score
import argparse
from utils import create_logger, save_model
import os
import time
from tqdm import tqdm
import json
import itertools

os.environ["TOKENIZERS_PARALLELISM"] = "True"

root_path = os.path.abspath(os.path.dirname(__file__))
logger = create_logger(log_path=root_path + "/logs/train_w.log")


def train(args, dataloader, dev_dataloader, joint, optimizer, device):
    joint.train()
    best_dev_f1 = 0.0
    weights = json.load(open(root_path + "/datasets/label_weight.json", 'r', encoding="utf-8"))
    for epoch in range(args.epochs):
        logger.info("training epoch:%s" % epoch)
        f1, loss = train_epoch(args, dataloader, joint, optimizer, device, weights)
        logger.info("train  epoch:%s  avg f1:%.4f  avg loss:%.4f" % (epoch, f1, loss))
        dev_f1, dev_loss = eval(args, dev_dataloader, joint, device)
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            logger.info("best epoch:%s  best dev f1:%.4f" % (epoch, best_dev_f1))
        logger.info("save epoch:%s model path:%s" % (epoch, args.save_model_path))
        save_model(joint, epoch, optimizer, dev_loss, dev_f1, args.save_model_path)
        pass


def train_epoch(args, dataloader, model, optimizer, device, weights):
    model.train()
    total_f1 = 0.0
    total_loss = 0.0
    total_step = 0
    num_batch = dataloader.__len__()
    pbar = tqdm(dataloader, unit='batch', ncols=100)
    pbar.set_description("training ")
    for step, (batch, task_name) in enumerate(dataloader):
        model.to(device)
        input_ids, attention_mask, token_type_ids, labels = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)
        logits = model(input_ids=input_ids,
                       token_type_ids=token_type_ids,
                       attention_mask=attention_mask,
                       task_name=task_name.upper())

        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)

        if args.gradient_accumulation_steps > 1:
            loss /= args.gradient_accumulation_steps
        loss.backward()  # 损失反向传播，计算梯度
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()  # 梯度更新
            optimizer.zero_grad()  # 梯度清零

        total_loss += loss.item()

        def compute_score(label, prob):
            true_label = label.detach().cpu()
            pre_label = torch.max(prob.detach(), dim=-1)[1].cpu()
            f1 = f1_score(true_label, pre_label, average='macro')
            return f1

        f1 = compute_score(labels, logits)
        total_f1 += f1
        total_step += 1
        pbar.set_description(
            "train  step:%s  task_name-:%s  f1:%.4f loss:%.4f" % (total_step, task_name, f1, loss.item()))
        # if total_step % 1000 == 0:
        #     logger.info("train  step:%s  avg f1:%.4f  avg loss:%.4f  tnews f1:%.4f  ocnli_f1:%.4f  ocemotion f1:%.4f" %
        #                 (total_step, f1, loss.item(), tnews_f1, ocnli_f1, ocemotion_f1))

    return total_f1 / total_step, total_loss / total_step


def eval(args, dataloader, model, device):
    model.eval()
    total_loss = 0.0
    tnews_true_labels = []
    tnews_pre_labels = []
    ocnli_true_labels = []
    ocnli_pre_labels = []
    ocemotion_true_labels = []
    ocemotion_pre_labels = []

    tnews_dataloader = dataloader["tnews"]
    ocnli_dataloader = dataloader["ocnli"]
    ocemotion_dataloader = dataloader["ocemotion"]

    total_step = 0
    for task_name, loader in zip(("tnews", "ocnli", "ocemotion"),
                                 (tnews_dataloader, ocnli_dataloader, ocemotion_dataloader)):
        for step, batch in enumerate(loader):
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logits = model(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask,
                               task_name=task_name.upper())
                loss_fn = torch.nn.CrossEntropyLoss()
                loss = loss_fn(logits, labels)
            if task_name == "tnews":
                tnews_pre_label = torch.max(logits.detach(), dim=-1)[1].cpu().numpy()
                tnews_true_labels.extend(labels.detach().cpu().numpy())
                tnews_pre_labels.extend(tnews_pre_label)
            elif task_name == "ocnli":
                ocnli_pre_label = torch.max(logits.detach(), dim=-1)[1].cpu().numpy()
                ocnli_true_labels.extend(labels.detach().cpu().numpy())
                ocnli_pre_labels.extend(ocnli_pre_label)

            elif task_name == "ocemotion":
                ocemotion_pre_label = torch.max(logits.detach(), dim=-1)[1].cpu().numpy()
                ocemotion_true_labels.extend(labels.detach().cpu().numpy())
                ocemotion_pre_labels.extend(ocemotion_pre_label)
            else:
                raise ValueError('no task name')

            total_loss += loss.item()
            total_step += 1

    tnews_f1 = f1_score(tnews_true_labels, tnews_pre_labels, average='macro')
    ocnli_f1 = f1_score(ocnli_true_labels, ocnli_pre_labels, average='macro')
    ocemotion_f1 = f1_score(ocemotion_true_labels, ocemotion_pre_labels, average='macro')
    f1 = (tnews_f1 + ocnli_f1 + ocemotion_f1) / 3
    logger.info('eval  avg loss:%.4f  avg_f1:%.4f  tnews f1:%.4f  ocnli_f1:%.4f  ocemotion f1:%.4f' %
                (total_loss / total_step, f1, tnews_f1, ocnli_f1, ocemotion_f1))
    return f1, total_loss / total_step


def gen_train_dataset(args, tokenizer, config):
    tnews_train_dataset = DataPrecessForSentence(tokenizer, config.tnews_train_path, args.max_length)
    ocnli_train_dataset = DataPrecessForSentence(tokenizer, config.ocnli_train_path, args.max_length)
    emotion_train_dataset = DataPrecessForSentence(tokenizer, config.emotion_train_path, args.max_length)

    dataloader = {}
    tnews_dataloader = DataLoader(tnews_train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
    dataloader['tnews'] = tnews_dataloader
    ocnli_dataloader = DataLoader(ocnli_train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True)
    dataloader['ocnli'] = ocnli_dataloader
    emotion_dataloader = DataLoader(emotion_train_dataset,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    num_workers=2,
                                    drop_last=True)
    dataloader['ocemotion'] = emotion_dataloader

    return dataloader


def gen_dev_dataset(args, tokenizer, config):
    tnews_dev_dataset = DataPrecessForSentence(tokenizer, config.tnews_dev_path, args.max_length)
    ocnli_dev_dataset = DataPrecessForSentence(tokenizer, config.ocnli_dev_path, args.max_length)
    emotion_dev_dataset = DataPrecessForSentence(tokenizer, config.emotion_dev_path, args.max_length)

    dataloader = {}
    tnews_dataloader = DataLoader(tnews_dev_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=2)
    dataloader['tnews'] = tnews_dataloader
    ocnli_dataloader = DataLoader(ocnli_dev_dataset,
                                  batch_size=64,
                                  shuffle=True,
                                  num_workers=2)
    dataloader['ocnli'] = ocnli_dataloader
    emotion_dataloader = DataLoader(emotion_dev_dataset,
                                    batch_size=64,
                                    shuffle=True,
                                    num_workers=2)
    dataloader['ocemotion'] = emotion_dataloader

    return dataloader


def main(args):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("load bert")
    bert = BertModel.from_pretrained(args.model_path)
    bert.to(device)
    logger.info("load tokenizer")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    logger.info("load dataloader")
    train_dataloader = gen_train_dataset(args, tokenizer, config)
    train_multidataloader = MultitaskDataloader(train_dataloader)
    dev_dataloader = gen_dev_dataset(args, tokenizer, config)
    model = BertModel.from_pretrained(args.model_path)
    joint = JointModel(model)
    logger.info("load optimizer")
    optimizer = torch.optim.Adam(joint.parameters(), lr=args.lr)
    joint.to(device)
    logger.info("start training")
    train(args, train_multidataloader, dev_dataloader, joint, optimizer, device)

    eval(args, dev_dataloader, joint, device)
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="text classifier")
    parser.add_argument('--run_mode', dest="run_mode", action="store", default="train",
                        help="Running mode: train or eval")
    parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=32, help="")
    parser.add_argument("--max_length", dest="max_length", action="store", type=int, default=200, help="")
    parser.add_argument("--task_name", dest="task_name", action="store", default='tnews,ocnli,ocemotion', help="")
    parser.add_argument("--epochs", dest="epochs", action="store", type=int, default=5, help="")
    parser.add_argument("--lr", dest="lr", action="store", type=float, default=0.0001, help="")
    parser.add_argument("--save_model_path", dest="save_model_path", action="store", default="save_w_s",
                        help="")
    parser.add_argument("--model_path", dest="model_path", action="store", default="model/w_s/",
                        help="pretrained model")
    parser.add_argument("--log_path", dest="log_path", action="store", default="logs", help="")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", action="store",
                        type=int, default=8, help="")
    args = parser.parse_args()
    main(args)
