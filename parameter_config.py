#-*- coding: utf-8 -*-
import torch

class ParameterConfig():
    def __init__(self):
        # 判断是否使用GPU（1.电脑里必须有显卡；2.必须安装cuda版本的pytorch）
        # 下载cuda版本的pytorch链接：https://pytorch.org/get-started/previous-versions/
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 词典路径：在vocab文件夹里面
        self.vocab_path = './vocab/vocab.txt'
        # 训练文件路径
        self.train_path = './data/medical_train.pkl'
        # 验证数据文件路径
        self.valid_path = './data/medical_train.pkl'
        # 模型配置文件
        self.config_json = './config/config.json'
        # 模型保存路径
        self.save_model_path = './save_model'
        # 如果你有预训练模型就写上路径（我们本次没有直接运用GPT2它预训练好的模型，而是仅只用了该模型的框架）
        self.pretrained_model = './save_model/epoch97'
        # 保存对话语料
        self.save_corpus_path = './vocab'
        # 保存对话语料
        self.save_samples_path = 'sample'
        # 忽略一些字符：句子需要长度补齐，针对补的部分，没有意义，所以一般不进行梯度更新
        self.ignore_index = -100
        # 历史对话句子的长度
        self.max_history_len = 10# "dialogue history的最大长度"
        # 每一个完整对话的句子最大长度
        self.max_len = 50  # '每个utterance的最大长度,超过指定长度则进行截断,默认25'
        # self.repetition_penalty = 10.0 # "重复惩罚参数，若生成的对话重复性较高，可适当提高该参数"和蒸馏温度处理方法差不多
        # self.topk = 4 #'最高k选1。默认8'
        # self.batch_size = 8 #一个批次几个样本
        # self.epochs = 4 # 训练几轮
        # self.loss_step = 1 # 多少步汇报一次loss
        # 1. 重复惩罚（原来10.0太高了，导致生成太少）
        self.repetition_penalty = 1.2  # 改为1.2，防止重复但不过度抑制

        # 2. Top-k采样（原来4太小，生成太保守）
        self.topk = 40  # 改为40，增加多样性

        # 3. 新增：温度参数（控制随机性）
        self.temperature = 0.85  # 0.7-0.9之间，值越大越随机

        # 4. 新增：Top-p采样（配合top-k使用）
        self.topp = 0.9  # 核采样，保留概率累积和90%的token

        # 5. 训练轮次（你已经训练3轮，再训练1-2轮即可）

        # 6. 生成长度（对话时用）
        self.max_len = 100  # 生成回复的最大长度

        # 7. 历史对话长度（保持上下文）
        self.max_history_len = 3
        # self.epochs = 20  # 从1轮增加到10轮
        # self.lr = 1e-5  # 学习率稍微调低一点
        self.epochs = 1  # 从1轮增加到20轮
        self.lr = 2e-5  # 稍微提高学习率
        self.batch_size = 8
        # eps，为了增加数值计算的稳定性而加到分母里的项，其为了防止在实现中除以零
        self.eps = 1.0e-09
        self.max_grad_norm = 2.0
        self.gradient_accumulation_steps = 1
        # 默认.warmup_steps = 4000
        self.warmup_steps = 100 # 使用Warmup预热学习率的方式,即先用最初的小学习率训练，然后每个step增大一点点，直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），采用最初设置的学习率进行训练（注：预热学习率完成后的训练过程，学习率是衰减的），有助于使模型收敛速度变快，效果更佳。


if __name__ == '__main__':
    pc = ParameterConfig()
    print(pc.train_path)
    print(pc.device)
    print(torch.cuda.device_count())