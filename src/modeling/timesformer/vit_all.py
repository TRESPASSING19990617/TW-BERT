# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np

import torch.utils
import torch.utils.checkpoint

from src.modeling.timesformer.vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from src.modeling.timesformer.helpers import load_pretrained, load_pretrained_kinetics, load_pretrained_imagenet, \
    load_pretrained_CLIP_ViT
from src.modeling.timesformer.vit_utils import DropPath, to_2tuple, trunc_normal_

from src.modeling.xbert import BertAttention

# from .build import MODEL_REGISTRY
from torch import einsum
from einops import rearrange, reduce, repeat

import src.utils.grad_ckpt as grad_ckpt
from src.utils.logger import LOGGER, TB_LOGGER, add_log_to_file, RunningMeter


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'CLIP_ViT-B/16': _cfg(
        url='https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


###########################################
############# differential topK ###########
###########################################
class PerturbedTopK(nn.Module):
    def __init__(self, k: int, num_samples: int = 500, sigma: float = 0.05):
        super().__init__()
        self.num_samples = num_samples
        self.sigma = sigma
        self.k = k

    def __call__(self, x):
        return PerturbedTopKFuntion.apply(x, self.k, self.num_samples, self.sigma)


class PerturbedTopKFuntion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k: int, num_samples: int = 500, sigma: float = 0.05):
        # input here is scores with (bs, num_patches)
        b, d = x.shape
        noise = torch.normal(mean=0.0, std=1.0, size=(b, num_samples, d)).to(dtype=x.dtype, device=x.device)
        #print(x, torch.topk(x, k=k, dim=-1, sorted=False).indices)
        perturbed_x = x.unsqueeze(1) + noise * sigma  # b, nS, d
        topk_results = torch.topk(perturbed_x, k=k, dim=-1, sorted=False)
        indices = topk_results.indices  # b, nS, k
        indices = torch.sort(indices, dim=-1).values  # b, nS, k

        perturbed_output = F.one_hot(indices, num_classes=d).float()  # b, nS, k, d
        indicators = perturbed_output.mean(dim=1)  # b, k, d

        # context for backward
        ctx.k = k
        ctx.num_samples = num_samples
        ctx.sigma = sigma

        ctx.perturbed_output = perturbed_output
        ctx.noise = noise

        return indicators

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return tuple([None] * 5)

        noise_gradient = ctx.noise
        expected_gradient = (
                torch.einsum("bnkd,bnd->bkd", ctx.perturbed_output, noise_gradient)
                / ctx.num_samples
                / ctx.sigma
        )
        #print(expected_gradient.shape)
        grad_input = torch.einsum("bkd,bkd->bd", grad_output, expected_gradient)
        return (grad_input,) + tuple([None] * 5)


###########################################
############# differential topK ###########
###########################################

class PredictorLG(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, embed_dim=768):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim , bias=False),
            nn.GELU(),
            # nn.Linear(embed_dim // 2, embed_dim // 4, bias=False),
            # nn.GELU(),
            nn.Linear(embed_dim, 1, bias=False),
            # nn.Tanh()
            # nn.Sigmoid()
            nn.Softmax(dim=-2)
            # nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, y):
        '''
        x: shape (bs*n_length, num_tokens, hid_dim)
        '''
        x = self.in_conv(x) #b t m
        B, N, C = x.size()
        local_x = x[:, :, :]
        # print("global_x.shape: ", global_x.shape)
        x = torch.cat([local_x, y.expand(-1, N, -1)], dim=-1) #b t 2m
        #print(x.shape, self.out_conv(x).shape)
        return self.out_conv(x)


class VisualFrameSelection(nn.Module):
    def __init__(self, max_frames, embed_dim=768, topk=3):
        super().__init__()
        self.max_frames = max_frames
        self.score_predictor = PredictorLG(embed_dim=embed_dim)
        self.topk_selector = PerturbedTopK(topk)

    def forward(self, x, y, training=True):
        '''
        x: input embed, shape is (bs, length*Ntokens, hid_dim)
        use cls token as global representation
        prob = Tanh(MLP(x))
        '''

        B, L, D = x.shape #b (n t) m
        N = L // self.max_frames # 1 + (h w)
        x_cls = rearrange(x, 'b (n t) m -> b t n m', b=B, t=self.max_frames)[:, :, 0, :] #b t m
        pred_score = self.score_predictor(x_cls, y).squeeze(2)  # b t

        topk_indicator = self.topk_selector(pred_score)  # b k t

        x = rearrange(x, 'b (n t) m -> b t (n m)', b=B, t=self.max_frames)
        selected_frame_feature = torch.einsum("bkl,bld->bkd", topk_indicator, x)

        output = rearrange(selected_frame_feature, 'b k (n m) -> b (n k) m', b=B, m=D)


        return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            qkv = x.reshape(B, N, self.num_heads, C //
                            self.num_heads).permute(0, 2, 1, 3)
            q, k, v = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
            x = self.proj(x)
            x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, layer_num, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time',
                 use_grad_checkpointing=False):
        super().__init__()
        self.attention_type = attention_type
        assert (attention_type in ['divided_space_time',
                                   'space_only', 'joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        # drop path
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # [dxli]
        self.layer_num = layer_num
        self.use_grad_checkpointing = use_grad_checkpointing

    def forward(self, x, B, T, W):
        num_spatial_tokens = x.size(1) // T -1 #b (n t) m
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            # Temporal
            xt = x
            xt = rearrange(xt, 'b (n t) m -> (b n) t m',
                           b=B, t=T)

            if self.use_grad_checkpointing:
                # temporal_attn_out = torch.utils.checkpoint.checkpoint(self.temporal_attn, self.temporal_norm1(xt))
                temporal_attn_out = grad_ckpt.CheckpointFunction.apply(self.temporal_attn, 1, self.temporal_norm1(xt))
            else:
                temporal_attn_out = self.temporal_attn(self.temporal_norm1(xt))
                # res_temporal = self.drop_path(
                #     self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = self.drop_path(temporal_attn_out)

            res_temporal = rearrange(
                res_temporal, '(b n) t m -> b (n t) m', b=B, t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x + res_temporal

            # Spatial
            '''init_cls_token = x[:, 0, :].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(
                cls_token, 'b t m -> (b t) m', b=B, t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',
                           b=B, h=H, w=W, t=T)
            xs = torch.cat((cls_token, xs), 1)'''

            xs = xt
            xs = rearrange(xs, 'b (n t) m -> (b t) n m', b=B, t=T)

            # [origial]
            # res_spatial = self.drop_path(self.attn(self.norm1(xs)))
            if self.use_grad_checkpointing:
                spatial_attn_out = grad_ckpt.CheckpointFunction.apply(self.attn, 1, self.norm1(xs))
            else:
                # spatial_attn_out = torch.utils.checkpoint.checkpoint(self.attn, self.norm1(xs))
                spatial_attn_out = self.attn(self.norm1(xs))
            res_spatial = self.drop_path(spatial_attn_out)

            # Taking care of CLS token
            '''cls_token = res_spatial[:, 0, :]
            cls_token = rearrange(cls_token, '(b t) m -> b t m', b=B, t=T)
            # averaging for every frame
            cls_token = torch.mean(cls_token, 1, True)
            res_spatial = res_spatial[:, 1:, :]
            res_spatial = rearrange(
                res_spatial, '(b t) (h w) m -> b (h w t) m', b=B, h=H, w=W, t=T)
            res = res_spatial'''
            res = rearrange(res_spatial, '(b t) n m -> b (n t) m', b=B, t=T)
            x = xt
            x = res + x #b (n t) m

            # Mlp
            #x = torch.cat((init_cls_token, x), 1) + \
                #torch.cat((cls_token, res), 1)

            x_res = x #b (n t) m

            x = self.norm2(x)
            # x = x + self.drop_path(self.mlp(self.norm2(x)))

            # MLP
            # [origial]
            # x = x_res + self.drop_path(self.mlp(x))
            if self.use_grad_checkpointing:
                # mlp_out = torch.utils.checkpoint.checkpoint(self.mlp, x)
                mlp_out = grad_ckpt.CheckpointFunction.apply(self.mlp, 1, x)
            else:
                mlp_out = self.mlp(x)

            x = x_res + self.drop_path(mlp_out)
            return x #b (n t) m


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
                      (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=16, select_layers=[5, 10], select_ratio=0.5,
                 attention_type='divided_space_time', dropout=0.,
                 cross_attention_config=None, use_grad_checkpointing=False):
        super().__init__()

        self.attention_type = attention_type
        self.visual_token_selector1 = VisualFrameSelection(num_frames, embed_dim, topk=int(num_frames * select_ratio))
        self.visual_token_selector2 = VisualFrameSelection(int(num_frames * select_ratio), embed_dim, topk=int(int(num_frames * select_ratio) * select_ratio))
        self.select_layers = select_layers
        self.select_ratio = select_ratio
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        # Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(layer_num=i, use_grad_checkpointing=use_grad_checkpointing,
                  dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                  attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        # initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                        nn.init.constant_(m.temporal_fc.weight, 0)
                        nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, y, return_all_tokens=False):
        B = x.shape[0]
        x, T, W = self.patch_embed(x) #(b t) n m
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1) #(b t) 1 m
        x = torch.cat((cls_tokens, x), dim=1)

        # resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(
                other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        # Time Embeddings
        if self.attention_type != 'space_only':
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            # Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(
                    time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)#b (1+H*W T) m

        # Attention blocks
        for blk in self.blocks:
            if (blk.layer_num+1)<self.select_layers[0]:
                x = blk(x, B, T, W)
            if (blk.layer_num+1)==self.select_layers[0]:
                x = blk(x, B, T, W)
                x = self.visual_token_selector1(x, y)
            if (blk.layer_num+1)>self.select_layers[0] and (blk.layer_num+1)<self.select_layers[1]:
                x = blk(x, B, int(T * self.select_ratio), W)
            if (blk.layer_num+1)==self.select_layers[1]:
                x = blk(x, B, int(T * self.select_ratio), W)
                
                x_cls = rearrange(x, 'b (n t) m -> b t n m', b=B, t=int(T * self.select_ratio))[:, :, 0, :] #b t m
                pred_score = self.visual_token_selector2.score_predictor(x_cls, y).squeeze(2) #b t
                #print(pred_score)
                scores = torch.topk(pred_score, k=int(int(T * self.select_ratio) * self.select_ratio), dim=-1, sorted=True).values #b k
                #print(scores)
                scores = torch.sum(scores, 1, False) #b
                #print(scores)
                
                x = self.visual_token_selector2(x, y)
            if (blk.layer_num+1)>self.select_layers[1]:
                x = blk(x, B, int(int(T * self.select_ratio) * self.select_ratio), W)

        # Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m', b=B, t=T)
            x = torch.mean(x, 1)  # averaging predictions for every frame

        x = self.norm(x)

        if return_all_tokens:
            return x, scores
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict


class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained = True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES,
                                       patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
                                       qkv_bias=True, norm_layer=partial(
                nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                       num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE,
                                       **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * \
                           (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model = cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3),
                            filter_fn=_conv_filter,
                            img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches,
                            attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x


class TimeSformer(nn.Module):
    def __init__(self, model_cfg, input_format='BGR', cross_attention_config=None, **kwargs):
        super(TimeSformer, self).__init__()

        self.config_file = str(model_cfg)

        # model-specific configurations
        self.select_layers = model_cfg['select_layers']
        self.select_ratio = model_cfg['select_ratio']
        self.img_size = model_cfg['img_size']
        self.patch_size = model_cfg['patch_size']
        self.num_frames = model_cfg['num_frm']
        self.attn_drop_rate = model_cfg['attn_drop_rate']
        self.drop_path_rate = model_cfg['drop_path_rate']
        self.drop_rate = model_cfg['drop_rate']
        self.use_pooling = model_cfg['use_maxpooling']
        self.use_grad_ckpt = model_cfg['gradient_checkpointing']

        self.attention_type = 'divided_space_time'

        LOGGER.info(
            f'Initializing TimeSformer with img_size={self.img_size}, patch_size={self.patch_size}, num_frames={self.num_frames}')

        # will be ignored when loading official pretrained ckpt
        self.num_classes = 400

        self.input_format = input_format
        assert input_format == "RGB", "Official TimeSformer uses RGB input."

        self.model = VisionTransformer(img_size=self.img_size,
                                       num_classes=self.num_classes,
                                       patch_size=self.patch_size,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       mlp_ratio=4,
                                       qkv_bias=True,
                                       norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                       drop_rate=self.drop_rate,
                                       attn_drop_rate=self.attn_drop_rate,
                                       drop_path_rate=self.drop_path_rate,
                                       num_frames=self.num_frames,
                                       select_layers=self.select_layers,
                                       select_ratio=self.select_ratio,
                                       attention_type=self.attention_type,
                                       cross_attention_config=cross_attention_config,
                                       use_grad_checkpointing=self.use_grad_ckpt,
                                       **kwargs
                                       )

        if self.use_pooling:
            self.maxpool_kernel_size = model_cfg['maxpool_kernel_size']
            self.maxpooling = torch.nn.MaxPool2d(kernel_size=self.maxpool_kernel_size)

        self.model.default_cfg = default_cfgs['vit_base_patch' + str(self.patch_size) + '_224']
        self.num_patches = (self.img_size // self.patch_size) * (self.img_size // self.patch_size)
        self.meanpooling = torch.nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        x = self.model(x)
        return x

    def forward_features(self, x, y, return_all_tokens=True, pooling='spatial'):
        b, c, t, h, w = x.shape
        # print(x.shape)

        x, scores = self.model.forward_features(x, y, return_all_tokens=return_all_tokens) # b (n t) m
        # print(x.shape)

        ## apply pooling
        W = H = self.img_size // self.patch_size
        T = int(int(self.num_frames * self.select_ratio) * self.select_ratio)

        x = rearrange(x, 'b (n t) m -> b n t m', t=T)
        cls_tokens = torch.mean(x[:, 0, :, :], dim=1).unsqueeze(1) #b 1 m
        other_tokens = x[:, 1:, :, :] # b (h w) t m

        x = rearrange(other_tokens, 'b (h w) t m -> b m t h w', h=H, w=W, t=T)

        assert pooling in ['temporal', 'spatial', 'none'], 'Invalid pooling type {}'.format(pooling)
        if pooling == 'temporal':
            x = torch.mean(x, dim=1)
            x = torch.cat((cls_tokens, x), dim=1)
        elif pooling == 'spatial':  # spatial pooling
            # x = torch.max(x, dim=2)[0]
            # x = torch.mean(x, dim=2)
            x = self.meanpooling(x)
            x = rearrange(x, 'b m t h w -> b (t h w) m')
            x = torch.cat((cls_tokens, x), dim=1)
        elif pooling == 'none':
            cls_tokens_repeat = cls_tokens.unsqueeze(1).repeat(1, T, 1, 1)
            x = torch.cat((cls_tokens_repeat, x), dim=2)
        else:
            raise NotImplementedError('Unsupported pooling type {}'.format(pooling))

        #print(x.shape)
        return x, scores

    def _get_pooled_features(self, x):
        b, t, h, w, c = x.shape

        # x = rarrange(x.transpose(2, 4).transpose(3, 4), 'b t h w c -> (b t c) h w')
        x = rearrange(x, 'b t h w c -> (b t c) h w')
        x = self.maxpooling(x)
        x = rearrange(x, '(b t c) h w -> b (t h w) c', b=b, t=t)

        return x

    def load_state_dict(self, pretrained_ckpt_path):
        LOGGER.info('Loading TimeSformer checkpoints from {}'.format(pretrained_ckpt_path))

        if pretrained_ckpt_path == "vit_base_patch16_224":
            load_ckpt_func = load_pretrained_imagenet
        elif "CLIP_ViT" in pretrained_ckpt_path:
            load_ckpt_func = load_pretrained_CLIP_ViT
        else:
            load_ckpt_func = load_pretrained

        load_ckpt_func(self.model,
                       num_classes=self.model.num_classes,
                       in_chans=3,
                       filter_fn=_conv_filter,
                       img_size=self.img_size,
                       num_frames=self.num_frames,
                       num_patches=self.num_patches,
                       attention_type=self.attention_type,
                       pretrained_model=pretrained_ckpt_path
                       )