import os
import joblib
from transformers import BertTokenizer, BertConfig
from gen_datasets import GenDatasets
from torch.utils.data import DataLoader
from bert_model import JointModel
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from utils import load_model, create_logger
import transformers
import json
from transformers import BertModel

# 随机数固定
seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
# 日志记录
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
root_path = os.path.abspath(os.path.dirname(__file__))
logger = create_logger(log_path=root_path + "/infer.log")


def infer_model(args, model, dataloader, id2label, master_gpu_id, use_cuda, task_name=None):
    model.eval()
    dataloader = dataloader[task_name][args.run_mode]
    pbar = tqdm(dataloader, unit='batch', ncols=100)
    pbar.set_description('test step %s' % task_name)
    pre = []
    for step, (input_ids, token_type_ids, attention_mask) in enumerate(pbar):
        input_ids = input_ids.cuda(master_gpu_id) if use_cuda else input_ids
        token_type_ids = token_type_ids.cuda(master_gpu_id) if use_cuda else token_type_ids
        attention_mask = attention_mask.cuda(master_gpu_id) if use_cuda else attention_mask

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                           task_name=task_name.upper())
            pre_label = torch.max(logits.detach(), dim=1)[1].cpu().numpy()
            pre.extend(pre_label)
    ids = list(range(dataloader.dataset.__len__()))
    assert len(ids) == len(pre), "数据长度不一致"
    if not os.path.exists(args.save_result_path):
        os.mkdir(args.save_result_path)
    with open(args.save_result_path + '/{}_predict.json'.format(task_name), 'w', encoding='utf-8') as writer:
        for id, label in zip(ids, pre):
            label = id2label[int(label)]
            text = json.dumps({"id": str(id), "label": str(label)})
            writer.write(text + "\n")


def main(args):
    logger.info('Load Modeling')
    bert = BertModel.from_pretrained(args.model_path)
    model = JointModel(bert)
    optimizer = None
    model, optim = load_model(model, optimizer, args.save_model_path)
    #     logger.info('Load Modeling:%s' % model)
    logger.info('Gpu or Cpu')
    use_cuda = args.gpu_ids != '-1'
    if len(args.gpu_ids) == 1 and use_cuda:
        master_gpu_id = int(args.gpu_ids)
        model = model.cuda(master_gpu_id) if use_cuda else model
    elif use_cuda:
        gpu_ids = [int(each) for each in args.gpu_ids.split(",")]
        master_gpu_id = gpu_ids[0]
        model = model.cuda(master_gpu_id)
        logger.info("Start multi-gpu dataparallel training/evaluating...")
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
    g = GenDatasets(tokenizer, batch_size=args.batch_size, sampler=False)
    tnews_id2label, ocnli_id2label, ocemotion_id2label = g.tnews_id2label, g.ocnli_id2label, g.ocemotion_id2label
    tnews_test_dataloader, ocnli_test_dataloader, ocemotion_test_dataloader = \
        g.tnews_test_dataloader, g.ocnli_test_dataloader, g.ocemotion_test_dataloader
    dataloader['tnews']["test"] = tnews_test_dataloader
    dataloader['ocnli']["test"] = ocnli_test_dataloader
    dataloader['ocemotion']["test"] = ocemotion_test_dataloader
    logger.info("Load Testing dataloader Done, Total training line: %s", tnews_test_dataloader.__len__())
    logger.info("Load Testing dataloader Done, Total training line: %s", ocnli_test_dataloader.__len__())
    logger.info("Load Testing dataloader Done, Total training line: %s", ocemotion_test_dataloader.__len__())
    #
    infer_model(args, model, dataloader, tnews_id2label, master_gpu_id, use_cuda, task_name='tnews')
    infer_model(args, model, dataloader, ocnli_id2label, master_gpu_id, use_cuda, task_name='ocnli')
    infer_model(args, model, dataloader, ocemotion_id2label, master_gpu_id, use_cuda, task_name='ocemotion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="text classifier")
    parser.add_argument('--run_mode', dest="run_mode", action="store", default="test",
                        help="Running mode: test")
    parser.add_argument("--batch_size", dest="batch_size", action="store", type=int, default=64, help="")
    parser.add_argument("--save_model_path", dest="save_model_path", action="store", default="save_w", help="")
    parser.add_argument("--model_path", dest="model_path", action="store", default="model/w/",
                        help="pretrained model")
    parser.add_argument("--save_result_path", dest="save_result_path", action="store", default="result", help="")
    parser.add_argument("--gpu_ids", dest="gpu_ids", action="store", default="0",
                        help="Device ids of used gpus, split by ',' , IF -1 then no gpu")
    args = parser.parse_args()
    main(args)
