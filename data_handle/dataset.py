from  torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self,input_ids,max_len):
        self.input_ids = input_ids
        self.max_len = max_len

    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, id):
        input_id = self.input_ids[id]
        input_id = input_id[:self.max_len]
        input_id = torch.tensor(input_id,dtype=torch.long)
        return input_id

if __name__ == '__main__':
    import pickle
    input_ids = pickle.load(open(r'D:\黑马python+ai\53-黑马程序员-2025年python人工智能开发 V6.5\阶段九\009 大模型开发基础与项目-V5.0-AI版\03-code\练习版基于gpt的问诊机器人\data\medical_valid_1.pkl','rb'))
    dataset = MyDataset(input_ids,10)
    # 获取第三个样本
    print(dataset.__getitem__(3))