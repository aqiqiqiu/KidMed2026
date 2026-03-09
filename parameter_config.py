# -*- coding: utf-8 -*-
import torch


class ParameterConfig():
    def __init__(self):
        # 判断是否使用GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 使用Qwen2.5模型
        self.model_name = "Qwen/Qwen2.5-1.5B-Instruct"  # 1.5B版本，更轻量
        # 或者用 "Qwen/Qwen2.5-3B-Instruct" 如果你有8GB以上显存

        # 本地模型保存路径（微调后保存）
        self.save_model_path = r'./saved_models/qwen_medical'

        # 训练数据路径
        self.train_path = r'./data/medical_train.pkl'
        self.valid_path = r'./data/medical_train.pkl'

        # 训练参数
        self.epochs = 3  # 少轮次，因为预训练模型已经很强
        self.lr = 2e-5
        self.batch_size = 2  # 根据显存调整，1.5B模型可以设4-8
        self.gradient_accumulation_steps = 4  # 梯度累积，相当于batch_size=8

        self.max_len = 512  # 最大长度
        self.ignore_index = -100

        # 生成参数
        self.max_history_len = 3
        self.temperature = 0.7
        self.topk = 40
        self.topp = 0.9

        # 其他训练参数
        self.warmup_steps = 100
        self.max_grad_norm = 1.0
        self.eps = 1.0e-08