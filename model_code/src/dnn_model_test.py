# 用训练好的 DNN 模型对不同品牌的数据进行测试，并计算不同 top-k 数量下的正例比例，最后将结果输出到控制台并保存到文件中
# 1、get_data(ak_config)：同 dnn_model_train.py 中的 get_data 函数，用于获取数据。
# 2、get_data_test(ak_config, brand_id)：从配置文件 ak_config 中获取相关信息，连接到 ODPS 项目，读取指定品牌 brand_id 的测试数据，进行数据处理（如删除非特征列、分离特征和标签），最后返回测试集的特征与标签的 NumPy 数组。
# 3、calculate_top_k_ratio(predictions, labels, top_k_list)：
#   参数：predictions 是模型的预测结果，labels 是真实标签，top_k_list 是要计算的不同 top-k 数量的列表。
#   功能：将预测结果和真实标签组合成 DataFrame 并按预测分数降序排序，计算总正例数，然后针对 top_k_list 中的每个 k，计算前 k 个样本中的正例数占总正例数的比例，最后返回一个字典，键为 k，值为对应的比例。
# 4、my_collate_fn(batch)：同 dnn_model_train.py 中的 my_collate_fn 函数，用于处理一批样本的数据。





import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import os
from odps import ODPS
from odps.df import DataFrame
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

from dataset.dnn_dataset import MyDataset,MyPriorDataset
from model.dnn_model import MyModel
from config.ak_config import config as ak_config
from config.dnn_config import config as dnn_config
from utils.get_data import get_data,get_data_test
from utils.get_data import calculate_top_k_ratio
from utils.get_data import my_collate_fn

brands = ['b47686','b56508','b62063','b78739']
for brand_id in brands:
    #brand_id='b56508'
    model_path = './models/focal2/model_epoch_1_27999.pth'    #第27999步的auc的value是0.8608（即A点）
    # 需要计算top数量
    top_k_list = [1000, 3000, 5000, 10000, 50000]

    test_feature_numpy,test_label = get_data_test(ak_config, brand_id)
    dataset_test = MyPriorDataset(test_feature_numpy, test_label, dnn_config)

    dataloader_test = DataLoader(dataset_test, batch_size=dnn_config["batch_size"], shuffle=False, collate_fn=my_collate_fn)

    model = MyModel(dnn_config).to('cpu')
    model.load_state_dict(torch.load(model_path))
    model.to('cpu')
    model.eval()
    test_preds = []
    test_targets = []
    for data, target in dataloader_test:
        output  = model(data)
        test_preds.extend(output.sigmoid().squeeze().tolist())
        test_targets.extend(target.squeeze().tolist())

    # 计算top k的正例比例
    ratios = calculate_top_k_ratio(test_preds, test_targets, top_k_list)
    # 输出结果
    for k, ratio in ratios.items():
        print(f"Top {k} ratio of positive labels: {ratio:.4f}")
    # 如果需要保存结果到文件
    with open(f'models_{brand_id}_top_results_focal.txt', 'w') as f:
        for k, ratio in ratios.items():
            f.write(f"Top {k} ratio of positive labels: {ratio:.4f}\n")

