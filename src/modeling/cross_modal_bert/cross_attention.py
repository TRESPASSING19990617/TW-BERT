import matplotlib.pyplot as plt
import cv2
#import seaborn as sns
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from src.modeling.cross_modal_bert.cross_modules import *

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def Cross_attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        seq_len=query.size()[-2]
        b = torch.from_numpy(np.zeros(seq_len, dtype=np.int32).reshape(seq_len,1)).cuda()
        #print((mask|b).shape, scores.shape)
        if len(query.shape)==5:
            scores = scores.masked_fill((mask|b).unsqueeze(1).unsqueeze(2) == 0, -1e9)
        else:
            scores = scores.masked_fill((mask|b).unsqueeze(1) == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class Cross_MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(Cross_MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 8)
        self.attn1 = None
        self.attn2 = None
        self.attn3 = None
        self.dropout = nn.Dropout(p=dropout)
        #self.norm = LayerNorm(d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-12)
        
    def forward(self, all_token, mask, T, H, W):
        if mask is not None:
            # Same mask applied to all h heads.
            mask2 = mask[:,:-T*H*W-1+T].unsqueeze(1)
            mask3 = mask[:,:-T*H*W+H*W].unsqueeze(1)
        nbatches = all_token.size(0)
        txt_len = all_token.size(1)-T*H*W-1
         
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        # query,key,value shape: (nbatches, h, T, seq_len, d_k)
        txt=all_token[:,:-T*H*W-1,:]
        video=all_token[:,-T*H*W-1:,:]
        
        query1, key1, value1 = [l(x) for l, x in zip(self.linears, (txt, video[:,1:T*H*W+1,:], video[:,1:T*H*W+1,:]))]
        query1 = query1.view(nbatches, -1, self.h, self.d_k).transpose(1, 2).unsqueeze(2)#(nbatches, h, 1, txt_len, d_k)
        key1 = key1.view(nbatches, T, -1, self.h, self.d_k).permute(0, 3, 1, 2, 4)#(nbatches, h, T, video_len, d_k)
        value1 = value1.view(nbatches, T, -1, self.h, self.d_k).permute(0, 3, 1, 2, 4) #(nbatches, h, T, video_len, d_k)
        
        # 2) Apply attention on all the projected vectors in batch. 
        x1, self.attn1 = Cross_attention(query1, key1, value1, None, dropout=self.dropout)#(nbatches, h, T, txt_len, d_k)
        x1 = x1.transpose(1, 3).contiguous().view(nbatches, -1, T, self.h * self.d_k)
        x1 = self.norm(self.linears[7](x1))
        
        #################################################################################
        query2 = self.linears[3](txt)
        query2 = query2.view(nbatches, -1, self.h, self.d_k).unsqueeze(3)#(nbatches, txt_len, h, 1, d_k)
        
        x1 = torch.cat([txt.unsqueeze(1).repeat(1,txt_len,1,1), x1], dim=2)
        key2, value2 = self.linears[4](x1), self.linears[5](x1)
        key2 = key2.view(nbatches, -1, T+txt_len, self.h, self.d_k).transpose(2, 3)#(nbatches, txt_len, h, txt_len+T, d_k)
        value2 = value2.view(nbatches, -1, T+txt_len, self.h, self.d_k).transpose(2, 3)#(nbatches, txt_len, h, txt_len+T, d_k)
        
        # 4) Apply attention on all the projected vectors in batch.
        #print(query2.shape, key2.shape, value2.shape, mask2.shape)
        x2, self.attn2 = Cross_attention(query2, key2, value2, mask2, dropout=self.dropout)
        x2 = x2.squeeze(3).view(nbatches, -1, self.h * self.d_k)
        x2 = self.linears[6](x2)
        
        #################################################################################
        query3, key3_txt, value3_txt = self.linears[0](video), self.linears[1](txt), self.linears[2](txt)
        key3_cls, value3_cls = self.linears[1](video[:,0,:]), self.linears[2](video[:,0,:])
        
        
        query3_no_cls = query3[:,1:T*H*W+1,:].view(nbatches, T, -1, self.h, self.d_k).permute(0, 3, 1, 2, 4)#(nbatches, h, T, video_len, d_k)
        query3_cls = query3[:,0,:].view(nbatches, 1, self.h, self.d_k).transpose(1, 2).unsqueeze(2)#(nbatches, h, T, cls, d_k)
        query3 = torch.cat([query3_cls.repeat(1,1,T,1,1), query3_no_cls], dim=3)#(nbatches, h, T, 1+video_len, d_k)
        
        
        key3_cls = key3_cls.view(nbatches, 1, self.h, self.d_k).transpose(1, 2).unsqueeze(2)#(nbatches, h, T, cls, d_k)
        value3_cls = value3_cls.view(nbatches, 1, self.h, self.d_k).transpose(1, 2).unsqueeze(2)#(nbatches, h, T, cls, d_k)
        key3_txt = key3_txt.view(nbatches, -1, self.h, self.d_k).transpose(1, 2).unsqueeze(2)#(nbatches, h, 1, txt_len, d_k)
        value3_txt = value3_txt.view(nbatches, -1, self.h, self.d_k).transpose(1, 2).unsqueeze(2) #(nbatches, h, 1, txt_len, d_k)
        key3 = torch.cat([key3_txt.repeat(1,1,T,1,1), key1, key3_cls.repeat(1,1,T,1,1)], dim=3)#(nbatches, h, T, 1+video_len+txt_len, d_k)
        value3 = torch.cat([value3_txt.repeat(1,1,T,1,1), value1, value3_cls.repeat(1,1,T,1,1)], dim=3)#(nbatches, h, T, 1+video_len+txt_len, d_k)
        
        x3, self.attn3 = Cross_attention(query3, key3, value3, mask3, dropout=self.dropout)#(nbatches, h, T, 1+video_len, d_k)
        x3 = x3.permute(0, 2, 3, 1, 4).contiguous().view(nbatches, T, -1, self.h * self.d_k)
        x3_no_cls = x3[:,:,1:,:].reshape(nbatches, -1, self.h * self.d_k)
        x3_cls = torch.mean(x3[:,:,0,:], dim=1).unsqueeze(1)
        x3 = self.linears[6](torch.cat([x3_cls, x3_no_cls], dim=1))
        
        #print(T, H, W, torch.cat([x2, x3], dim=1).shape)
        return torch.cat([x2, x3], dim=1)

