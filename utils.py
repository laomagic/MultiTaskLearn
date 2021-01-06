from torch.optim import Adam
from collections import OrderedDict
import torch
import os
import re
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import logging
from logging import handlers
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import f1_score
import json



def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)



def save_model(model, epoch, optim, loss, f1, path):
    """保存模型"""
    if not os.path.exists(path):
        os.mkdir(path)
    named_param = OrderedDict()
    named_param['net'] = model.state_dict()
    named_param['optim'] = optim.state_dict()
    named_param['epoch'] = epoch
    named_param['loss'] = loss
    named_param['f1'] = f1
    model_name = "Epoch_" + str(epoch) + "_F1_" + str(round(f1, 4)) + ".bin"
    model_file = os.path.join(path, model_name)
    torch.save(named_param, model_file)
    print("dumped model file to:%s", model_file)


def load_model(model, optim, saved_model_path, model_file=None):
    """加载模型"""
    if model_file == None:
        files = os.listdir(saved_model_path)
        # max_idx = 0
        max_f1 = 0
        max_fname = ''
        for fname in files:
            # idx = re.sub('Epoch_|\.bin', '', fname)
            f1 = re.sub('Epoch_[0-9]+_F1_|.bin', '', fname)
            if float(f1) > max_f1:
                max_f1 = float(f1)
                max_fname = fname
        model_file = max_fname
    model_file = os.path.join(saved_model_path, model_file)
    named_param = torch.load(model_file, map_location="cpu")
    model_weight = named_param['net']
    optim_weight = named_param['optim']
    model.load_state_dict(model_weight)
    if optim:
        optim.load_state_dict(optim_weight)
    print('loaded saved model file:%s', model_file)
    return model, optim


def compute_score(prob, label):
    true_label = label.detach().cpu()
    pre_label = torch.max(prob.detach(), dim=-1)[1].cpu()
    # acc_score = accuracy_score(true_label, pre_label)
    f1 = f1_score(true_label, pre_label, average='macro')
    return f1


def compute_loss(batch_output, loss_weight=None, weight=False):
    loss = 0.0
    try:
        tnews_weight, ocnli_weight, ocemotion_weight = loss_weight
    except:
        tnews_weight, ocnli_weight, ocemotion_weight = None, None, None
    tnews_loss = nn.CrossEntropyLoss(weight=tnews_weight)
    ocnli_loss = nn.CrossEntropyLoss(weight=ocnli_weight)
    ocemotion_loss = nn.CrossEntropyLoss(weight=ocemotion_weight)
    loss_fn = nn.CrossEntropyLoss()

    for task_name in batch_output:
        if weight:
            if task_name == "TNEWS":
                loss += tnews_loss(*batch_output[task_name])
            elif task_name == "OCNLI":
                loss += ocnli_loss(*batch_output[task_name])
            elif task_name == "OCEMOTION":
                loss += ocemotion_loss(*batch_output[task_name])
            else:
                raise ValueError("%s is error" % task_name)
        else:
            loss += loss_fn(*batch_output[task_name])
    return loss


def create_logger(log_path):
    """
    日志的创建
    :param log_path:
    :return:
    """
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    logger = logging.getLogger(log_path)
    fmt = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    format_str = logging.Formatter(fmt)  # 设置日志格式
    logger.setLevel(level_relations.get('info'))  # 设置日志级别
    sh = logging.StreamHandler()  # 往屏幕上输出
    sh.setFormatter(format_str)  # 设置屏幕上显示的格式
    th = handlers.TimedRotatingFileHandler(
        filename=log_path, when='D', backupCount=3,
        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
    th.setFormatter(format_str)  # 设置文件里写入的格式
    logger.addHandler(sh)  # 把对象加到logger里
    logger.addHandler(th)

    return logger


class CELoss(nn.Module):
    def __init__(self, class_num, alpha=None, use_alpha=False, size_average=True):
        super(CELoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * prob.log().double() * target_.double()
        else:
            batch_loss = - prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        # print(prob[0],target[0],target_[0],batch_loss[0])
        # print('--')

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, use_alpha=False, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average

    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1, self.class_num))
        prob = prob.clamp(min=0.0001, max=1.0)

        target_ = torch.zeros(target.size(0), self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(), 1.)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1 - prob,
                                                           self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1 - prob, self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss