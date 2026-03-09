#导入rnn_utils模块，用于处理可变长度序列的填充和排序
import torch.nn.utils.rnn as rnn_utils
from click.core import batch

#导入dataset和dataLoader模块，用用户加载和处理数据集
from torch.utils.data import Dataset,DataLoader

import torch
import pickle

from 练习版基于gpt的问诊机器人.data_handle.dataset import MyDataset


def load_dataset(train_pkl,valid_pkl):
    #加载训练集和验证集
# ：param train_path:训练数据集路径
# ：return:训练数据集和验证数据集
    with open(train_pkl,'rb') as f:
        train_list = pickle.load(f)  #从文件中加载输入列表
    with open(valid_pkl,'rb') as f:
        valid_list = pickle.load(f)  #从文件中加载输入列表
    #划分训练集与验证集

    #创建训练集对象
    train_dataset = MyDataset(train_list,200)
    #创建验证集
    valid_dataset = MyDataset(valid_list,200)
    #返回训练数据集和验证数据集
    return train_dataset,valid_dataset

def collate_fn(batch):
    #是他们的长度一致
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)
    label = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids,label
def get_loader(train_path,valid_path):
    train_dataset,valid_dataset = load_dataset(train_path,valid_path)
    train_dataloader = DataLoader(train_dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn,
                            drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,
                            batch_size=4,
                            shuffle=True,
                            collate_fn=collate_fn,
                            drop_last=True)
    return train_dataloader,valid_dataloader

if __name__ == '__main__':
    pkl = r'D:\黑马python+ai\53-黑马程序员-2025年python人工智能开发 V6.5\阶段九\009 大模型开发基础与项目-V5.0-AI版\03-code\练习版基于gpt的问诊机器人\data\medical_valid_1.pkl'
    # train_dataset,valid_dataset = load_dataset(pkl,pkl)
    train_dataloader,valid_dataloader= get_loader(pkl,pkl)
    for data in valid_dataloader:
        print(data)
        break