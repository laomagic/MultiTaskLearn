"""对训练完成的模型进行量化"""
import torch
from transformers import BertModel
from bert_model import JointModel
import os
from utils import load_model
torch.backends.quantized.engine = 'qnnpack'
path = "/home/serverai/yanxubin/tianchi_pre/save_w/"
print('before quantization Size (MB):', os.path.getsize(path) / 1e6)
bert = BertModel.from_pretrained("/home/serverai/yanxubin/tianchi_pre/model/w/")
model = JointModel(bert)
optim = None
model, _ = load_model(model, optim, path)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

    
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8)
print('after quantization Size (MB):')
print_size_of_model(quantized_model)
