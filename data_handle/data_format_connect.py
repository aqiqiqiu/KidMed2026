# import pandas as pd
# #监控进度
# from tqdm import tqdm
# def readcsv2txt():
#     data = pd.read_excel('')
#     print(data.head())
#     #转换成list
#     data_list = data.values.tolist()
#     for data in tqdm(data_list):
#         try:
#             question = data[2]
#             answer = data[3]
#             str1 = question + '\n' + answer
#             with open('./data/train.txt', 'a') as f:
#                 f.write(str + '\n\n')
#         except:
#             continue
# readcsv2txt()


