# -*-coding: utf-8 -*-
# @Time    : 2024/12/20 16:10
# @File    : llm.py
# @Software: PyCharm

'''
该代码主要是生成句子的embedding向量，
embedding大模型采用智源的开源embedding大模型  https://github.com/FlagOpen/FlagEmbedding

关于模型的输出维度，可见https://baijiahao.baidu.com/s?id=1773837931893405085&wfr=spider&for=pc

https://github.com/FlagOpen/FlagEmbedding/blob/master/Tutorials/1_Embedding/1.1_Intro%26Inference.ipynb

BAAI/bge-base-en-v1.5  模型输出的维度为768维
'''

import os
import datetime
import time
import argparse
import pandas as pd
import numpy as np
import torch

os.system("pip install transformers")


from transformers import AutoTokenizer, AutoModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dim', help='数据的输出维度', type=int, default=768)
    parser.add_argument('--data_output', help='Output file path', type=str, default='')
    parser.add_argument('--data_input', help='map_feature_file', type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    # BAAI/bge-base-en-v1.5预训练模型下载地址: https://huggingface.co/BAAI/bge-base-en-v1.5/tree/main
    tokenizer = AutoTokenizer.from_pretrained("./BAAI/bge-base-en-v1.5")
    model = AutoModel.from_pretrained("./BAAI/bge-base-en-v1.5")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("模型加载完成")
    '''
    # 测试用例
    sentences = '我们的祖国是花园'
    inputs = tokenizer(sentences, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # 获取最后一层的隐藏状态
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    print("embeddings=", embeddings)
    print("embeddings=", embeddings.shape)
    '''
    prompt_words = 'Please output the embedding for the following sentence: This scenario is a social recommendation system in a game; the player\'s ID is {}; this player has {} friends; the top 50 friends\' IDs with the highest scores obtained by the Personalized PageRank algorithm (separated by commas) are {}; the Personalized PageRank scores of these top 50 friends (separated by commas) are {}; as of today, the number of consecutive days these 50 friends have not logged into the game, i.e., the churn days (separated by commas) are {}; the average Personalized PageRank score of these 50 friends is {}; the average churn days of these 50 friends is {}.'
    # 读取数据
    path = args.data_input.split(',')[0]
    input_files = sorted([file for file in os.listdir(path) if file.find("part-") != -1])
    count = -1
    for file in input_files:
        start_time = time.time()
        count += 1
        # 读取训练数据
        data = pd.read_csv(os.path.join(path, file), sep=';', header=None).astype(object)
        n_samples, _ = data.shape
        result = np.zeros((n_samples, args.output_dim)).astype(str)
        result[:, 0] = data.iloc[:, 0].values
        for i in range(n_samples):
            sentences = prompt_words.format(data.iloc[i, 0], data.iloc[i, 1], data.iloc[i, 2], data.iloc[i, 3],
                                                 data.iloc[i, 4], data.iloc[i, 5], data.iloc[i, 6])
            # 编码输入文本，获取嵌入
            inputs = tokenizer(sentences, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # 获取最后一层的隐藏状态
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            result[i, 1:] = embeddings.numpy().astype(str)[0]
        end_time = time.time()
        output_file = os.path.join(args.data_output, 'pred_{}.csv'.format(count))

        # 使用 numpy.savetxt 写入 CSV 文件
        with open(output_file, mode="a") as resultfile:
            # 写入数据
            np.savetxt(resultfile, result, delimiter=',', fmt='%s')  # 使用 %s 以支持字符串和数字
        print("第{}个数据文件已经写入完成,写入数据的行数{} 耗时:{} {}".format(count, n_samples, end_time - start_time,
                                                                              datetime.datetime.now()))
