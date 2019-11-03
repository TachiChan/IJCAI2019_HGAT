import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from gat_layers import BatchMultiHeadGraphAttention


class BatchGAT(nn.Module):
    def __init__(
            self, word_feature, interaction_item,
            interaction_word, use_user_feature,
            use_item_feature, use_word_feature,
            n_units=[16,16], n_heads=[8,8,1],
            item_dim=200, user_dim=200,
            dropout=0.1, attn_dropout=0.0, fine_tune=False,
            instance_normalization=False):
        super(BatchGAT, self).__init__()
        self.n_layer = len(n_units)
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
        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            self.layer_stack.append(
                    BatchMultiHeadGraphAttention(
                        n_heads[i], f_in=f_in,
                        f_out=n_units[i + 1], attn_dropout=attn_dropout))

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

        bs, n = adj.size()[:2]
        for i, gat_layer in enumerate(self.layer_stack):
            x = gat_layer(x, adj) 
            if i + 1 == self.n_layer:
                x = x.mean(dim=1)  
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        return F.log_softmax(x, dim=-1), x[:, -1, :]
