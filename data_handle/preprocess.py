 #导入分词器
from transformers import BertTokenizerFast
#将数据保存为pkl文件方便下次读取
import pickle
#读取文件的进度条展示
from tqdm import tqdm

def preprocess(train_txt_path,train_pkl_path):
    #对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    #初始化tokenizer，使用BertTokenizerFast.从预训练的中文Bert模型（bert-base-chinese）创建一个tokenizer对象'''
    tokenizer = BertTokenizerFast(r'D:\黑马python+ai\53-黑马程序员-2025年python人工智能开发 V6.5\阶段九\009 大模型开发基础与项目-V5.0-AI版\03-code\练习版基于gpt的问诊机器人\vocab\vocab.txt')
    sep_id = tokenizer.sep_token_id     #获取分隔符的tokenid
    cls_id = tokenizer.cls_token_id     #获取起始符
    with open(train_txt_path,'rb') as f:
        data = f.read().decode('utf-8')


    if '\r\n' in data:
        data_list = data.split('\r\n\r\n')
    else:
        data_list = data.split('\n\n')

    dialogue_len = []   #记录所有对话thokenizer之后的长度，用于统计中位数与均值
    dialogue_list = []  #保存所有对话
    for id,dialogue in enumerate(tqdm(data_list)):
        if '\r\n' in dialogue:
            seq_list = dialogue.split('\r\n\r\n')
        else:
            seq_list = dialogue.split('\n\n')
        input_ids = [cls_id]
        for seq in seq_list:
            input_ids += tokenizer.encode(seq,add_special_tokens=False)
            input_ids += [sep_id]
        dialogue_list.append(input_ids)
        dialogue_len.append(len(input_ids))
    print(dialogue_len)
    with open(train_pkl_path,'wb') as f:
        pickle.dump(dialogue_list,f)

if __name__ == '__main__':
    # train_txt_path = r'D:\黑马python+ai\53-黑马程序员-2025年python人工智能开发 V6.5\阶段九\009 大模型开发基础与项目-V5.0-AI版\03-code\KidMed-Q2\data\medical_train.txt'
    train_txt_path = r'/KidMed-Q2\data\medical_valid.txt'
    train_pkl_path = r'/KidMed-Q2\data\medical_valid_1.pkl'
    preprocess(train_txt_path,train_pkl_path)