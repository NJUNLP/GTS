import copy
import math

import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SelfAttention(torch.nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()
        self.args = args
        self.linear_q = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim * 2)
        # self.linear_k = torch.nn.Linear(configs.BILSTM_DIM * 2, configs.BILSTM_DIM * 2)
        # self.linear_v = torch.nn.Linear(configs.BILSTM_DIM * 2, configs.BILSTM_DIM * 2)
        # self.w_query = torch.nn.Linear(configs.BILSTM_DIM * 2, 50)
        # self.w_value = torch.nn.Linear(configs.BILSTM_DIM * 2, 50)
        self.w_query = torch.nn.Linear(args.cnn_dim, 50)
        self.w_value = torch.nn.Linear(args.cnn_dim, 50)
        self.v = torch.nn.Linear(50, 1, bias=False)

    def forward(self, query, value, mask):
        # attention_states = self.linear_q(query)
        # attention_states_T = self.linear_k(values)
        attention_states = query
        attention_states_T = value
        attention_states_T = attention_states_T.permute([0, 2, 1])

        weights=torch.bmm(attention_states, attention_states_T)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))    #   mask掉每行后面的列
        attention = F.softmax(weights,dim=2)

        # value=self.linear_v(states)
        merged=torch.bmm(attention, value)
        merged=merged * mask.unsqueeze(2).float().expand_as(merged)

        return merged

    def forward_perceptron(self, query, value, mask):
        attention_states = query
        attention_states = self.w_query(attention_states)
        attention_states = attention_states.unsqueeze(2).expand(-1,-1,attention_states.shape[1], -1)

        attention_states_T = value
        attention_states_T = self.w_value(attention_states_T)
        attention_states_T = attention_states_T.unsqueeze(2).expand(-1,-1,attention_states_T.shape[1], -1)
        attention_states_T = attention_states_T.permute([0, 2, 1, 3])

        weights = torch.tanh(attention_states+attention_states_T)
        weights = self.v(weights).squeeze(3)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))    #   mask掉每行后面的列
        attention = F.softmax(weights,dim=2)

        merged = torch.bmm(attention, value)
        merged = merged * mask.unsqueeze(2).float().expand_as(merged)
        return merged
