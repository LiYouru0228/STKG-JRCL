import os
import imp
import re
import pickle
import datetime
import math
import copy

import torch
from torch import nn
import torch.nn.utils.rnn as rnn_utils
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() == True else 'cpu')
print("available device: {}".format(device))

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index).to(device)

def get_loss(y_pred, y_true):
    loss = torch.nn.BCELoss()
    return loss(y_pred, y_true)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0 

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn 
    
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, self.d_k * self.h), 3)
        self.final_linear = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1) 

        nbatches = query.size(0)
        input_dim = query.size(1)
        feature_dim = query.size(-1)
        
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))] 
        
       
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)

      
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        return self.final_linear(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-7):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        returned_value = sublayer(self.norm(x))
        return x + self.dropout(returned_value[0]) , returned_value[1]
    
class Expert(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

class RoMoE(nn.Module):
    def __init__(self, input_size, num_experts, experts_out, experts_hidden, channels):
        super(RoMoE, self).__init__()
        self.input_size = input_size
        self.num_experts = num_experts
        self.experts_out = experts_out
        self.experts_hidden = experts_hidden
        self.channels = channels

        self.softmax = nn.Softmax(dim=1)
        self.experts = nn.ModuleList([Expert(self.input_size, self.experts_out, self.experts_hidden) for i in range(self.num_experts)])
        self.w_gates = nn.ParameterList([nn.Parameter(torch.randn(input_size, int(num_experts/2)), requires_grad=True) for i in range(self.channels)])

    def forward(self, x, y, z):
        # add randly expert:
        experts_o1 = [e(x) for e in self.experts]
        experts_o2 = [e(y) for e in self.experts]
        experts_o3 = [e(z) for e in self.experts]
        
        experts_o1_tensor = torch.stack(random.sample(experts_o1, int(len(experts_o1)/2)))
        experts_o2_tensor = torch.stack(random.sample(experts_o2, int(len(experts_o2)/2)))
        experts_o3_tensor = torch.stack(random.sample(experts_o3, int(len(experts_o3)/2)))

        gates_o1 = self.softmax(x @ self.w_gates[0]) 
        gates_o2 = self.softmax(y @ self.w_gates[1]) 
        gates_o3 = self.softmax(z @ self.w_gates[2]) 

        refined_emb1 = gates_o1.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o1_tensor
        refined_emb2 = gates_o2.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o2_tensor
        refined_emb3 = gates_o3.t().unsqueeze(2).expand(-1, -1, self.experts_out) * experts_o3_tensor

        refined_embs = [refined_emb1, refined_emb2, refined_emb3]
        final_output = [torch.sum(refined_emb, dim=0) for refined_emb in refined_embs]
        final_output = torch.cat([final_output[0],final_output[1],final_output[2]],1)

        return final_output
    
class JRCL(nn.Module):
    def __init__(self, input_dim, hidden_dim, d_model, MHD_num_head, d_ff, output_dim, keep_prob=0.5):
        super(JRCL, self).__init__()

        # hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim 
        self.d_model = d_model
        self.MHD_num_head = MHD_num_head
        self.d_ff = d_ff
        self.output_dim = output_dim
        self.keep_prob = keep_prob

        # layers
        self.LSTMs = clones(nn.LSTM(self.input_dim*self.hidden_dim, self.hidden_dim, batch_first = True), 2)        
        self.MultiHeadedAttention = MultiHeadedAttention(self.MHD_num_head, self.d_model,dropout = 1 - self.keep_prob)
        self.SublayerConnection = SublayerConnection(self.hidden_dim, dropout = 1 - self.keep_prob)
        
        self.pair_proj_main = nn.Linear(2118, self.hidden_dim)
        self.RoMoE = RoMoE(self.hidden_dim, 4, 16, 16, 3)
        self.output0 = nn.Linear(16*3, self.hidden_dim)
        self.output1 = nn.Linear(self.hidden_dim, self.output_dim)
        self.output2 = nn.Linear(64*20, self.hidden_dim)
        
        self.dropout = nn.Dropout(p = 1 - self.keep_prob)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, input1, input2, each_epoch, step):
        batch_size = input.size(0)
        feature_dim = input2.size(1)
        assert(feature_dim == self.input_dim)
        assert(self.d_model % self.MHD_num_head == 0)

        # forward
        pair_info_embedd = self.tanh(self.pair_proj_main(input))
        paths_embeded_input = input1
        paths_embeded_output = []
        
        for i in range(input1.size(-1)):
            posi_input = self.dropout(paths_embeded_input[:,:,:,i]) 
            contexts = self.SublayerConnection(posi_input, lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None))
            paths_embeded_output.append(contexts[0].view(contexts[0].size(0), -1))
            
        paths_embeded_output = torch.stack(paths_embeded_output,dim=1)   
        paths_embeded_output = self.LSTMs[0](paths_embeded_output, (Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device),Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device)))[0][:,-1,:]        
        
        lbs_embeded_input = input2
        lbs_embeded_output = []
        
        for i in range(input2.size(-1)):
            posi_input = self.dropout(lbs_embeded_input[:,:,:,i]) 
            contexts = self.SublayerConnection(posi_input, lambda x: self.MultiHeadedAttention(posi_input, posi_input, posi_input, None))
            lbs_embeded_output.append(contexts[0].view(contexts[0].size(0), -1))
        
        lbs_embeded_output = torch.stack(lbs_embeded_output,dim=1)   
        lbs_embeded_output = self.LSTMs[1](lbs_embeded_output, (Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device),Variable(torch.zeros(batch_size, self.hidden_dim).unsqueeze(0)).to(device)))[0][:,-1,:]
        
        weighted_contexts = self.RoMoE(pair_info_embedd, paths_embeded_output, lbs_embeded_output)
        
        output = self.output1(self.relu(self.output0(weighted_contexts)))
        output = self.sigmoid(output)
          
        return output
