# 定义了一个名为 MyModel 的 DNN 神经网络模型：可以接收包含多个特征列的输入数据，经过嵌入、特征拼接和多层全连接变换后，输出一个预测值，可用于回归或二分类等任务。
# (1)_init__ 方法中，根据传入的配置字典 config 初始化模型的各层
# (2)forward 方法定义了模型的前向传播过程:
#     对每个特征列进行嵌入操作，并对嵌入结果按维度 1 求和。
#     将所有特征列的嵌入结果在维度 1 上拼接起来。
#     通过两个带有 ReLU 激活函数的全连接层进行非线性变换。
#     最后通过一个全连接层输出预测结果。

import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

class MyModel(nn.Module):
    def __init__(self, config):
        super(MyModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=self.config["num_embedding"], embedding_dim =self.config["embedding_dim"], padding_idx=0)
        self.fc1 = nn.Linear(self.config["embedding_dim"]*len(self.config["feature_col"]), 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,1)
        
    def forward(self, features):
        embedding_dict = {}
        for ff in self.config["feature_col"]:
            embedding_dict[ff] = torch.sum(self.embedding(features[ff]), dim=1)
        x = torch.cat([embedding_dict[ff] for ff in self.config["feature_col"]], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
