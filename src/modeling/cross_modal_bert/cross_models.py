import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.modeling.cross_modal_bert.cross_attention import *
from torch.nn import CrossEntropyLoss

    
class Cross_Encoder(nn.Module):
    def __init__(self, layer, N, d_model):
        super(Cross_Encoder, self).__init__()
        self.layers = clones(layer, N)
        #self.norm = LayerNorm(layer.size)
        self.norm = nn.LayerNorm(layer.size, eps=1e-12)

    def forward(self, inputs, mask, T=4, H=7, W=7):
        x = inputs
        for layer in self.layers:
            x = layer(x, mask, T, H, W)

        x = self.norm(x)
        return x


class Cross_EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(Cross_EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, T, H, W):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, mask, T, H, W))
        return self.sublayer[1](x, self.feed_forward)
