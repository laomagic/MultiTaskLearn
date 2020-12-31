import os
from transformers import BertTokenizer, BertModel
from joint_dataset import JDataset, collate_fn
from torch.utils.data import DataLoader
from bert_model import JointModel
import json
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from utils import save_model, load_model, create_logger, compute_loss, compute_score
from itertools import zip_longest

# 随机数固定
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
# 日志记录
root_path = os.path.abspath(os.path.dirname(__file__))
logger = create_logger(log_path=root_path + "/logs/train_ratio_w.log")


def train_model(args, dataloader, model, optimizer, epochs, master_gpu_id, use_cuda=False, gradient_accumulation_steps=1):
    logger.info("Start Training")
    total_epoch = 0
    for epoch in range(1, epochs + 1):
        logger.info("Training Epoch:{}/{}".format(epoch, epochs))
        train_epoch(model, optimizer, dataloader, master_gpu_id, gradient_accumulation_steps, use_cuda)
        logger.info("Eval" + "=" * 70)
        tnews_f1, tnews_loss = eval_model(args, model, dataloader, master_gpu_id, use_cuda, task_name="tnews")
        ocnli_f1, ocnli_loss = eval_model(args, model, dataloader, master_gpu_id, use_cuda, task_name='ocnli')
        emotion_f1, emotion_loss = eval_model(args, model, dataloader, master_gpu_id, use_cuda, task_name='ocemotion')
        f1 = (tnews_f1 + ocnli_f1 + emotion_f1) / 3
        loss = (tnews_loss + ocnli_loss + emotion_loss)
        total_epoch += epoch
        logger.info('eval epoch:%s  eval loss:%.4f  eval f1:%.4f' % (epoch, loss, f1))
        logger.info("Eval" + "=" * 70)
        # 保存模型
        save_model(model, epoch, optimizer, loss, f1, args.save_model_path)


def train_epoch(model, optimizer, dataloader, master_gpu_id, gradient_accumulation_steps, use_cuda):
    global global_step
    global_step = 0
    model.train()
    total_loss = 0.0
    total_f1 = 0.0

    weights = json.load(open('data/label_weights.json', 'r', encoding="utf-8"))
    tnews_weight = torch.tensor(weights["TNEWS"]).cuda(master_gpu_id) if use_cuda else torch.tensor(weights["TNEWS"])
    ocnli_weight = torch.tensor(weights["OCNLI"]).cuda(master_gpu_id) if use_cuda else torch.tensor(weights["OCNLI"])
    ocemotion_weight = torch.tensor(weights["OCEMOTION"]).cuda(master_gpu_id) if use_cuda else torch.tensor(
        weights["OCEMOTION"])
    loss_weight = (tnews_weight, ocnli_weight, ocemotion_weight)

    tnews_dataloader = dataloader["tnews"]["train"]
    ocnli_dataloader = dataloader["ocnli"]["train"]
    ocemotion_dataloader = dataloader["ocemotion"]["train"]
    pbar = tqdm(zip_longest(tnews_dataloader, ocnli_dataloader, ocemotion_dataloader), unit='batch', ncols=100)
    pbar.set_description('train step')
    task_names = [task.upper() for task in args.task_name.strip().split(',')]
    for step, batches in enumerate(pbar):
        batch_output = {}
        for task_name, batch in zip(task_names, batches):
            if batch is None:
                continue
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.cuda(master_gpu_id) if use_cuda else input_ids
            token_type_ids = token_type_ids.cuda(master_gpu_id) if use_cuda else token_type_ids
            attention_mask = attention_mask.cuda(master_gpu_id) if use_cuda else attention_mask
            labels = labels.cuda(master_gpu_id) if use_cuda else labels

            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, task_name=task_name)
            batch_output[task_name] = (logits, labels)  # 保存logits和labels到字典

        loss = compute_loss(batch_output, loss_weight=loss_weight, weight=True)  # 计算损失
        if gradient_accumulation_steps > 1:
            loss /= gradient_accumulation_steps

        loss.backward()  # 损失反向传播，计算梯度
        if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()  # 梯度更新
            optimizer.zero_grad()  # 梯度清零

        try:
            tnews_f1 = compute_score(*batch_output["TNEWS"])
        except:
            tnews_f1 = 0.0
            logger.info("step:%s TNEWS is not exist" % step)
        try:
            ocnli_f1 = compute_score(*batch_output["OCNLI"])
        except:
            ocnli_f1 = 0.
            logger.info("step:%s OCNLI is not exist" % step)
        try:
            ocemotion_f1 = compute_score(*batch_output["OCEMOTION"])
        except:
            ocemotion_f1 = 0.0
            logger.info("step:%s OCEMOTION is not exist" % step)

        f1 = tnews_f1 + ocnli_f1 + ocemotion_f1
        total_loss += loss.item()
        total_f1 += f1
        global_step += 1
        if global_step % 500 == 0:
            logger.info("training  step:%s  loss:%.4f  f1:%.4f  tnews_f1:%.4f  ocnli_f1:%.4f  ocemotion_f1:%.4f" %
                        (global_step, loss.item(), f1 / 3, tnews_f1, ocnli_f1, ocemotion_f1))


def eval_model(args, model, dataloader, master_gpu_id, use_cuda, task_name=None):
    model.eval()
    dataloader = dataloader[task_name]["dev"]
    total_loss = 0.0
    pre_all = []
    true_all = []
    num_batch = dataloader.__len__()
    pbar = tqdm(dataloader, unit='batch', ncols=100)
    pbar.set_description('eval step')
    for step, (input_ids, attention_mask, token_type_ids, labels) in enumerate(pbar):
        input_ids = input_ids.cuda(master_gpu_id) if use_cuda else input_ids
        attention_mask = attention_mask.cuda(master_gpu_id) if use_cuda else attention_mask
        token_type_ids = token_type_ids.cuda(master_gpu_id) if use_cuda else token_type_ids
        labels = labels.cuda(master_gpu_id) if use_cuda else labels

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, task_name=task_name.upper())
            loss = nn.CrossEntropyLoss()(logits, labels)
        true_label = labels.detach().cpu().numpy()
        pre_label = torch.max(logits.detach(), dim=1)[1].cpu().numpy()
        total_loss += loss.item()
        pre_all.extend(pre_label)
        true_all.extend(true_label)
    f1 = f1_score(true_all, pre_all, average='macro')
    logger.info('task name :%s eval total avg loss:%.4f  dev f1:%.4f' %
                (task_name, total_loss / num_batch, f1))
    return f1, loss.item()


def main(args):
    logger.info("train paramerts:%s" % args)
    logger.info('Load Modeling')
    model_name = args.model_path
    bert = BertModel.from_pretrained(model_name)
    if args.run_mode == "train":
        model = JointModel(bert)
    elif args.run_mode == "eval":
        model = JointModel(bert)
        optim = None
        load_model(model, optim, args.save_model_path, model_file=None)
    else:
        raise RuntimeError('Operation Mode not Legal')
    logger.info("model parameters:%s" % model)

    logger.info('Gpu or Cpu')
    use_cuda = args.gpu_ids != '-1'
    if len(args.gpu_ids) == 1 and use_cuda:
        master_gpu_id = int(args.gpu_ids)
        model = model.cuda(master_gpu_id) if use_cuda else model
    elif use_cuda:
        gpu_ids = [int(each) for each in args.gpu_ids.split(",")]
        master_gpu_id = gpu_ids[0]
        model = model.cuda(master_gpu_id)
        logging.info("Start multi-gpu dataparallel training/evaluating...")
        model = nn.DataParallel(model, device_ids=gpu_ids)
    else:
        master_gpu_id = None

    logger.info("Bert tokenizer")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)
    # 加载数据集
    logger.info("Dataset init")
    dataloader = {"tnews": {},
                  "ocnli": {},
                  "ocemotion": {}}
    if args.train_data:
        tnews_train_dataset = JDataset(tokenizer, task_name="TNEWS", mode="train")
        ocnli_train_dataset = JDataset(tokenizer, task_name="OCNLI", mode="train")
        ocemotion_train_dataset = JDataset(tokenizer, task_name="OCEMOTION", mode="train")
        tnews_length = tnews_train_dataset.__len__()
        ocnli_length = ocnli_train_dataset.__len__()
        ocemotion_length = ocemotion_train_dataset.__len__()
        length = tnews_length + ocnli_length + ocemotion_length
        ocnli_ratio = round((ocnli_length / length) * args.batch_size)
        ocemotion_ratio = round((ocemotion_length / length) * args.batch_size)
        tnews_ratio = args.batch_size - ocemotion_ratio - ocnli_ratio
        tnews_train_dataloader = DataLoader(tnews_train_dataset,
                                            collate_fn=collate_fn,
                                            batch_size=tnews_ratio,
                                            shuffle=True,
                                            num_workers=2)
        ocnli_train_dataloader = DataLoader(ocnli_train_dataset,
                                            collate_fn=collate_fn,
                                            batch_size=ocnli_ratio,
                                            shuffle=True,
                                            num_workers=2)
        ocemotion_train_dataloader = DataLoader(ocemotion_train_dataset,
                                                collate_fn=collate_fn,
                                                batch_size=ocemotion_ratio,
                                                shuffle=True,
                                                num_workers=2)

        dataloader['tnews']["train"] = tnews_train_dataloader
        dataloader['ocnli']["train"] = ocnli_train_dataloader
        dataloader['ocemotion']["train"] = ocemotion_train_dataloader
        logger.info("Load Training Dataloader Done, Total training line: %s", tnews_train_dataloader.__len__())
        logger.info("Load Training Dataloader Done, Total training line: %s", ocnli_train_dataloader.__len__())
        logger.info("Load Training Dataloader Done, Total training line: %s", ocemotion_train_dataloader.__len__())

    if args.eval_data:
        tnews_dev_dataset = JDataset(tokenizer, task_name="TNEWS", mode="dev")
        ocnli_dev_dataset = JDataset(tokenizer, task_name="OCNLI", mode="dev")
        ocemotion_dev_dataset = JDataset(tokenizer, task_name="OCEMOTION", mode="dev")
        tnews_dev_dataloader = DataLoader(tnews_dev_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                          shuffle=True)
        ocnli_dev_dataloader = DataLoader(ocnli_dev_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                          shuffle=True)
        ocemotion_dev_dataloader = DataLoader(ocemotion_dev_dataset, collate_fn=collate_fn, batch_size=args.batch_size,
                                              shuffle=True)
        dataloader['tnews']["dev"] = tnews_dev_dataloader
        dataloader['ocnli']["dev"] = ocnli_dev_dataloader
        dataloader['ocemotion']["dev"] = ocemotion_dev_dataloader
        logger.info("Load Evaling Dataloader Done, Total training line: %s", tnews_dev_dataloader.__len__())
        logger.info("Load Evaling Dataloader Done, Total training line: %s", ocnli_dev_dataloader.__len__())
        logger.info("Load Evaling Dataloader Done, Total training line: %s", ocemotion_dev_dataloader.__len__())

    if args.run_mode == "train":
        no_decay = ["bias", "gamma", "beta"]
        optimizer_parameters = [
            {"params": [p for name, p in model.named_parameters() \
                        if name not in no_decay], "weight_decay_rate": 0.01},
            {"params": [p for name, p in model.named_parameters() \
                        if name in no_decay], "weight_decay_rate": 0.0}
        ]
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(optimizer_parameters, lr=args.lr)
        train_model(args, dataloader,
                    model=model,
                    optimizer=optimizer,
                    epochs=args.epochs,
                    master_gpu_id=master_gpu_id,
                    use_cuda=use_cuda,
                    gradient_accumulation_steps=args.gradient_accumulation_steps)
    elif args.run_mode == "eval":
        tasks = args.task_name.strip().split(',')
        if isinstance(tasks, list):
            for task_name in tasks:
                eval_model(args, model, dataloader, master_gpu_id, use_cuda, task_name)
        else:
            raise RuntimeError('')
    else:
        raise RuntimeError('Mode not support:' + args.run_mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="joint model training")
    parser.add_argument('--train_data', dest="train_data", action="store", default=True, help="")
    parser.add_argument('--eval_data', dest="eval_data", action="store", default=True, help="")
    parser.add_argument('--test_data', dest="test_data", action="store", default=False, help="")
    parser.add_argument('--run_mode', dest="run_mode", action="store", default="train",
                        help="Running mode: train or eval")
    parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=64, help="")
    parser.add_argument("--task_name", dest="task_name", action="store", default='TNEWS,OCNLI,OCEMOTION', help="")
    parser.add_argument("--epochs", dest="epochs", action="store", type=int, default=5, help="")
    parser.add_argument("--lr", dest="lr", action="store", type=float, default=0.0001, help="")
    parser.add_argument("--save_model_path", dest="save_model_path", action="store", default="save_ratio_w",
                        help="")
    parser.add_argument("--model_path", dest="model_path", action="store", default="model/w_ratio_w/",
                        help="pretrained model")
    parser.add_argument("--log_path", dest="log_path", action="store", default="logs", help="")
    parser.add_argument("--gradient_accumulation_steps", dest="gradient_accumulation_steps", action="store",
                        type=int, default=16, help="")
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0",
                        help="Device ids of used gpus, split by ',' , IF -1 then no gpu")
    args = parser.parse_args()
    main(args)
