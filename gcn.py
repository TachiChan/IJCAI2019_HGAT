import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from gcn_layers import BatchGraphConvolution


class BatchGCN(nn.Module):
    def __init__(
            self, word_feature, interaction_item,
            interaction_word, use_user_feature,
            use_item_feature, use_word_feature,
            n_units=[128,128], 
            item_dim=200, user_dim=200,
            dropout=0.1,
            fine_tune=False, instance_normalization=False):
        super(BatchGCN, self).__init__()
        self.num_layer = len(n_units)
        self.dropout = dropout
        self.inst_norm = instance_normalization

        f_item, f_user = item_dim, user_dim
        n_units = [f_user] + n_units

        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(
                f_user, momentum=0.0, affine=True)

        self.use_user_feature = use_user_feature
        self.use_item_feature = use_item_feature
        self.use_word_feature = use_word_feature

        self.interaction_item = interaction_item
        self.interaction_word = interaction_word

        self.word_feature = nn.Embedding(
            word_feature.size(0), word_feature.size(1))
        self.word_feature.weight = nn.Parameter(word_feature)
        self.word_feature.weight.requires_grad = False

        if self.use_item_feature:
            self.w = Parameter(torch.Tensor(word_feature.size(1), f_user))
            self.bias = Parameter(torch.Tensor(f_user))
            self.attn = Parameter(torch.Tensor(f_user, 1))

            self.softmax = nn.Softmax(dim=-1)

            init.xavier_uniform_(self.w)
            init.constant_(self.bias, 0)
            init.xavier_uniform_(self.attn)
            
        if self.use_word_feature:
            self.w1 = Parameter(torch.Tensor(word_feature.size(1), f_item))
            self.bias1 = Parameter(torch.Tensor(f_item))
            self.attn1 = Parameter(torch.Tensor(f_item, 1))

            self.w2 = Parameter(torch.Tensor(f_item, f_user))
            self.bias2 = Parameter(torch.Tensor(f_user))
            self.attn2 = Parameter(torch.Tensor(f_user, 1))

            self.softmax = nn.Softmax(dim=-1)

            init.xavier_uniform_(self.w1)
            init.constant_(self.bias1, 0)
            init.xavier_uniform_(self.attn1)

            init.xavier_uniform_(self.w2)
            init.constant_(self.bias2, 0)
            init.xavier_uniform_(self.attn2)

        self.layer_stack = nn.ModuleList()
        for i in range(self.num_layer):
            self.layer_stack.append(
                    BatchGraphConvolution(n_units[i], n_units[i + 1]))

    def forward(self, vertices, adj):
        user_item = self.interaction_item[vertices]
        user_word = self.interaction_word[user_item]

        if self.use_word_feature:
            word_feature = self.word_feature(user_word)
            word_feature = torch.matmul(word_feature, self.w1) + self.bias1
            attn_coef1 = torch.matmul(torch.tanh(word_feature), self.attn1)
            attn_coef1 = self.softmax(attn_coef1).transpose(3, 4)
            item_feature = torch.matmul(attn_coef1, word_feature)
            item_feature = item_feature.squeeze(3)

            item_feature = torch.matmul(item_feature, self.w2) + self.bias2
            attn_coef2 = torch.matmul(torch.tanh(item_feature), self.attn2)
            attn_coef2 = self.softmax(attn_coef2).transpose(2, 3)
            x = torch.matmul(attn_coef2, item_feature)
            x = x.squeeze(2)

        if self.use_item_feature:
            word_feature = self.word_feature(user_word)
            item_feature = torch.mean(word_feature, dim=3)

            item_feature = torch.matmul(item_feature, self.w) + self.bias
            attn_coef = torch.matmul(torch.tanh(item_feature), self.attn)
            attn_coef = self.softmax(attn_coef).transpose(2, 3)
            x = torch.matmul(attn_coef, item_feature)
            x = x.squeeze(2)

        if self.use_user_feature:
            word_feature = self.word_feature(user_word)
            item_feature = torch.mean(word_feature, dim=3)
            x = torch.mean(item_feature, dim=2)

        if self.inst_norm:
            x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        for i, gcn_layer in enumerate(self.layer_stack):
            x = gcn_layer(x, adj)
            if i + 1 < self.num_layer:
                x = F.elu(x)
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1), x[:, -1, :]
