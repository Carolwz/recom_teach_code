import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

class LocalActivationUnit(nn.Module):
    def __init__(self, hidden_units):
        super(LocalActivationUnit, self).__init__()
        # 这里hidden_units表示embedding_dim(可从后面的输入看出)
        # 乘4是因为后面concat了4个特征
        self.fc1 = nn.Linear(hidden_units * 4, hidden_units) 
        self.fc2 = nn.Linear(hidden_units, 1)

    def forward(self, user_behaviors, target_item, mask):
        # user_behaviors shape: (batch_size, seq_len, hidden_units)
        # target_item shape: (batch_size, hidden_units)
        # mask shape: (batch_size, seq_len)

        #将target_item由(batch_size, hidden_units)扩充成与user_behaviors一样的shape：(batch_size, seq_len, hidden_units)
            # unsqueeze(i)在张量中位置i增加一个维度
            # squeeze(i)移除张量中最后一个维度
            # expand(-1, seq_len, -1)：将维度1扩展为seq_len（用户行为序列长度），-1表示维度保持不变。
        seq_len = user_behaviors.size(1)
        target_item = target_item.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Concatenate user behavior embeddings with target item embeddings
        interactions = torch.cat([user_behaviors, target_item, user_behaviors-target_item, user_behaviors*target_item], dim=-1)
        
        # Forward through two dense layers with activation
        x = torch.relu(self.fc1(interactions))
        attention_logits = self.fc2(x).squeeze(-1)  #移除张量(batch_size, seq_len, 1)中最后一个维度变成(batch_size, seq_len)
        
        # Apply mask to remove padding influence
            # 在计算注意力权重时，​忽略填充位置的影响。
            # 原理：Softmax特性，在计算Softmax时，-inf 的指数趋近于0，因此填充位置的注意力权重会趋近于0。
            # 功能：将 mask 中值为0的位置（填充位置）的注意力得分替换为 -inf（负无穷）。
        attention_logits = attention_logits.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax to compute attention weights
            # 对attention_logits中的第二个维度seq_len进行 Softmax 归一化，输出形状：(batch_size, seq_len)
            # 再将(batch_size, seq_len)扩充成(batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_logits, dim=1).unsqueeze(-1)
        ​
        
        # Compute weighted sum of user behavior embeddings to get user interests
        user_interests = torch.sum(attention_weights * user_behaviors, dim=1)
        return user_interests

class DinModel(nn.Module):
    def __init__(self, config):
        super(DinModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(num_embeddings=self.config["num_embedding"], embedding_dim =self.config["embedding_dim"], padding_idx=0)
        self.fc1 = nn.Linear(self.config["embedding_dim"]*len(self.config["feature_col"]), 512)
        self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(128,1)
        
        self.att = LocalActivationUnit(self.config["embedding_dim"])
        
    def forward(self, features, mask):
        embedding_dict = {}
        for ff in self.config["feature_col"]:
            if ff != 'pay_brand_seq':
                embedding_dict[ff] = torch.sum(self.embedding(features[ff]), dim=1)
        att_emb = self.att(self.embedding(features['pay_brand_seq']), embedding_dict['target_brand_id'], mask['pay_brand_seq'])
        x = torch.cat([embedding_dict[ff] for ff in self.config["feature_col"] if ff != 'pay_brand_seq'], dim=1)
        x = torch.cat([x,att_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
