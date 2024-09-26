from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import math
import pdb
from typing import Any, Union, List
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from pkg_resources import packaging
from typing import Optional, List
from transformer_module import TransformerDecoderLayer, TransformerDecoderLayer_womhsa
import random
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import os
from torchvision.ops.boxes import box_iou
from torchvision import transforms
import torch.utils.checkpoint as cp
import pickle

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)
        tgt2, attentions = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        # if torch.isnan(tgt2).any():
        #     print("tgt2", tgt2)
        #     print("tgt", tgt)
        #     print("memory", memory)
        #     print("query_pos", query_pos)
        #     print("key", self.with_pos_embed(memory, pos))
        #     pdb.set_trace()
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
   
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        # q = k = self.with_pos_embed(tgt2, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
        #                       key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt2 = self.norm2(tgt)
        tgt2, attentions = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderFusionLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, enable_cp=False):
        super().__init__()
        self.enable_cp = enable_cp
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.multihead_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.norm4 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, sup_memory=None,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # q = k = self.with_pos_embed(tgt, query_pos)
        # tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
        #                         key_padding_mask=tgt_key_padding_mask)[0]
        # tgt = tgt + self.dropout1(tgt2)
        # tgt = self.norm1(tgt)

        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = tgt + self.dropout2(tgt2)
        tgt2 = self.norm2(tgt2)

        tgt3 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(sup_memory, pos),
                                   value=sup_memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt3 = tgt + self.dropout2(tgt3)
        tgt3 = self.norm2(tgt3)

        tgt = tgt2 + tgt3
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, sup_memory=None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, sup_memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Extractor(nn.Module):
    def __init__(self, d_model, bottleneck, dropout,):
        super().__init__()
        '''
        update prior using F_vit

        K, V: F_vit
        Q: prior
        '''
        self.d_model = d_model
        self.down_size = bottleneck
        self.dropout = dropout  

        self.down_proj = nn.Linear(self.d_model, self.down_size)
        self.mhsa = TransformerDecoderLayer(self.down_size, 3, self.down_size*3,
                                                self.dropout, 'relu', normalize_before=True)
    
    def forward(self, prior, F_vit):
        # pdb.set_trace()
        query, mask = prior
        query = query.transpose(0,1) ## 18(#instance) x batchsize x down_size
        context = self.down_proj(F_vit) # 197 x batchsize x down_size
        new_prior = self.mhsa(tgt=query, memory=context,)
        return (new_prior.transpose(0,1), mask)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TextAdapter(nn.Module):
    def __init__(self,
                input_size,
                 dropout=0.1,
                 adapter_scalar="1.0",
                 adapter_num_layers=1,
                 mem_adpt_self = False, 
                 SA_only = False
                 ):
        super().__init__()
        self.n_embd = input_size
        self.down_size = 64
        self.scale = float(adapter_scalar)

        self.down_proj_mem = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_mem = nn.Linear(self.down_size, self.n_embd)
        self.adapter_num_layers = adapter_num_layers
        self.down_proj_prior = MLP(768, 128, self.down_size, 3)

        self.dropout = dropout
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down_proj_mem.weight, a=math.sqrt(5))
            nn.init.zeros_(self.up_proj_mem.weight)
            nn.init.zeros_(self.down_proj_mem.bias)
            nn.init.zeros_(self.up_proj_mem.bias)
    
        if SA_only is False:
            instance_decoder_layer = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)
        else:
            instance_decoder_layer = TransformerSALayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)            
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)
        self.mem_adpt_self = mem_adpt_self

    def forward(self, x, prior = None):
        tempa = self.down_proj_mem(x)
        if prior is None or self.mem_adpt_self is True:
            context = tempa.unsqueeze(0).transpose(0,1) ## 18(#instance) x batchsize x 64
        else:
            prior, mask = prior
            context = self.down_proj_prior(prior).transpose(0,1)
        tempa = self.non_linear_func(tempa) ## 197 x batchsize x 64
        # pdb.set_trace()
        for z, layer in enumerate(self.mhsa_layers):
            # pdb.set_trace()
            tempa = layer(tempa, context, tgt_mask=None,
                        memory_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=mask,
                        pos=None, query_pos=None)
        # pdb.set_trace()
        # tempa = self.transformer(tempa)
        
        up = self.up_proj_mem(tempa)
        output = (up * self.scale) + x
        return output



class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.1,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_num_layers=1,
                 multi_cross = False,
                 ):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck
        # import pdb
        # pdb.set_trace()
        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(d_model)*1e-9)
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.adapter_num_layers = adapter_num_layers

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)
        
        self.multi_cross = multi_cross
        # if multi_cross is True:
        #     instance_decoder_layer = TransformerDecoderFusionLayer(self.down_size, 2, self.down_size*2,
        #                                         self.dropout, 'relu', False)
        # else:
        instance_decoder_layer = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)
        # instance_decoder_norm = nn.LayerNorm(d_model)
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)
        # self.mhsa = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
        #                                         self.dropout, 'relu', False)

    def forward(self, x, prior=None, text_context = None):
        tempa = self.down_proj(x)
        tempa = self.non_linear_func(tempa) ## 197 x batchsize x 64  
        if prior is not None:
            context, mask = prior
            context = context.transpose(0,1) ## 18(#instance) x batchsize x 64
            if self.multi_cross is True:
                assert text_context is not None
                # pdb.set_trace()
                txt_prior, txt_mask = text_context
                #### need to do norm!!
                # text_context /= text_context.norm(dim=-1, keepdim=True)
                # text_context = text_context.unsqueeze(0)
                context = torch.cat((context, txt_prior.transpose(0,1)))
                mask = torch.cat((mask, txt_mask), dim=1)
            # pdb.set_trace()
            for z, layer in enumerate(self.mhsa_layers):
                # if self.multi_cross is True:
                #     tempa = layer(tempa, context, text_context, tgt_mask=None,
                #             memory_mask=None,
                #             tgt_key_padding_mask=None,
                #             memory_key_padding_mask=mask,
                #             pos=None, query_pos=None)
                # else:
                tempa = layer(tempa, context, tgt_mask=None,
                            memory_mask=None,
                            tgt_key_padding_mask=None,
                            memory_key_padding_mask=mask,
                            pos=None, query_pos=None)
        else:
            tempa = self.mhsa.forward_post(tempa, tempa, tgt_mask=None,
                           memory_mask=None,
                           tgt_key_padding_mask=None,
                           memory_key_padding_mask=None,
                           pos=None, query_pos=None)
        # for param in self.down_proj.parameters():
        #     if len(param.shape) == 2:
        #         print("tempa",param[0, 0])
        #     else:
        #         print("tempa",param[0])
        # for param in self.up_proj.parameters():
        #     if len(param.shape) == 2:
        #         print("up",param[0, 0])
        #     else:
        #         print("up",param[0])
        up = self.up_proj(tempa)
        output = up * self.scale
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.spacial_dim = spacial_dim
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x_old = x
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # add sptial position to B,C,H,W
        cls_pos = self.positional_embedding[0:1, :]
        # spatial_pos = F.interpolate(self.positional_embedding[1:,].reshape(1, self.spacial_dim, self.spacial_dim, self.embed_dim).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = self.positional_embedding[1:].reshape(self.spacial_dim, self.spacial_dim, self.embed_dim)[:H, :W]
        spatial_pos_old = spatial_pos
        spatial_pos = spatial_pos.reshape(-1, self.embed_dim)
        # spatial_pos = spatial_pos.reshape(self.embed_dim, H*W).permute(1, 0)
        positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
        
        try:
            x = x + positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        except:
            print(spatial_pos_old.shape,x_old.shape, H, W, B)
            pdb.set_trace()
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        x = x.permute(1, 2, 0)
        # pdb.set_trace()
        global_feat = x[:, :, 0]
        feature_map = x[:, :, 1:].reshape(B, -1, H, W)
        return global_feat, feature_map

        # return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        # pdb.set_trace()
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
    def init_weights(self, pretrained=None):
        # pdb.set_trace()
        pretrained = pretrained or self.pretrained
        
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.attnpool.positional_embedding.shape != state_dict[new_k].shape:
                            # pdb.set_trace()
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.attnpool.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 32
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            assert self.attnpool.positional_embedding.shape == state_dict[new_k].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in CLIPResNet')

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # pdb.set_trace()
        x_old = x
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        # pdb.set_trace()
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.attnpool(x)
        try:
            x_global, x_local = self.attnpool(x)
        except:
            print(x_old.shape)
        return x_global, x_local

        # return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, adapter: bool=False, adapter_num_layers: int=1, multi_cross: bool=False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        if adapter:
            self.adaptermlp = Adapter(None,  d_model=d_model , dropout=0.1, bottleneck=64,
                                    init_option='lora',
                                    adapter_scalar='learnable_scalar',
                                    adapter_num_layers=adapter_num_layers,
                                    multi_cross = multi_cross
                                    ) 
        self.adapter = adapter
        # self.adapter = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        '''
        x: L * bs * C, 
        prior[0]: bs * L' * C', padded prior knowledge
        prior[1]: bs * L' (mask of prior knowledge)
        ''' 
        if len(x) == 3:
            x, prior, text_context = x  
        elif len(x) == 2:
            x, prior = x
            text_context = None

        if self.adapter:
            adapt_x = self.adaptermlp(x, prior=prior, text_context = text_context)
            x = x + adapt_x

        tempa = self.attention(self.ln_1(x))

        x = x + tempa[0]  
        x = x + self.mlp(self.ln_2(x))  


        return (x, prior, text_context)
        # return (x,prior, tempa[1])


class ResidualAttentionBlock_MAPLE(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, adapter: bool=False, adapter_num_layers: int=1, multi_cross: bool=False,
                 text_layer = False, design_details = None, i=0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.text_layer = text_layer
        self.compound_prompt_nctx = design_details['maple_length']
        if adapter:
            self.adaptermlp = Adapter(None,  d_model=d_model , dropout=0.1, bottleneck=64,
                                    init_option='lora',
                                    adapter_scalar='learnable_scalar',
                                    adapter_num_layers=adapter_num_layers,
                                    multi_cross = multi_cross
                                    ) 
        self.adapter = adapter
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False
        if self.text_layer:
            self.txtcls_adapter = TextAdapter(768)
            # textemb_path_folder = "./checkpoints/text_emb"
            # textemb_path = os.path.join(textemb_path_folder, "gpt_text_emb.pkl")
            # with open(textemb_path, 'rb') as f:
            #     hoicls_txt = pickle.load(f)
            # self.hoicls_txt = torch.tensor(hoicls_txt)
        if 'init_txtcls_pt' in design_details.keys():
            self.init_txtcls_pt = design_details['init_txtcls_pt']
        else:
            self.init_txtcls_pt = False
        self.pt_begin_layer = design_details['pt_begin_layer']

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
        # return self.attn(x, x, x, need_weights=True, attn_mask=self.attn_mask)

    def forward(self, x: torch.Tensor):
        '''
        x: L * bs * C, 
        prior[0]: bs * L' * C', padded prior knowledge
        prior[1]: bs * L' (mask of prior knowledge)
        ''' 
        if len(x) == 3:
            x, prior, text_context = x  
        elif len(x) == 2:
            x, prior = x
            text_context = None
        
        compound_prompts_deeper = x[1]
        counter = x[2]
        if len(x) == 5 and self.text_layer:
            if isinstance(x[3], List):
                txtcls_ctx_pt = x[3]
                txtcls_feat = None
            else:
                txtcls_feat = x[3]
                txtcls_ctx_pt = None
            origin_ctx = x[4]
        else:
            txtcls_feat = None
            txtcls_ctx_pt = None
            origin_ctx = None
        x = x[0]
        
        if not self.first_layer:
            if origin_ctx is not None:
                origin_x = origin_ctx
            if (len(compound_prompts_deeper) > 0) or (len(txtcls_ctx_pt) > 0):

                if not self.text_layer:
                    if ((self.pt_begin_layer == 0) and (not (counter > len(compound_prompts_deeper) - 1))):
                        # Remove the outputs produced by learnable tokens of previous layer
                        prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                        # Create/configure learnable tokens of this layer
                        visual_context = compound_prompts_deeper[counter]  # extract the correct index
                        if len(visual_context.shape) == 3:
                            visual_context = visual_context.half()
                        else:
                            visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                        # Add the learnable tokens of this layer with the input, by replacing previous
                        # layer learnable tokens
                        x = torch.cat([prefix, visual_context.to(prefix.device)], dim=0)   
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                    elif self.pt_begin_layer > 0:
                        if (counter+2 >= self.pt_begin_layer and counter+2 <= len(compound_prompts_deeper) + self.pt_begin_layer -1):
                            temp_index = counter-self.pt_begin_layer+2
                            prefix = x[0:x.shape[0] - self.compound_prompt_nctx, :, :]
                            visual_context = compound_prompts_deeper[temp_index]  # extract the correct index
                            if len(visual_context.shape) == 3:
                                visual_context = visual_context.half()
                            else:
                                visual_context = visual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
                            x = torch.cat([prefix, visual_context.to(prefix.device)], dim=0)   
                            counter += 1
                                                    
                        elif counter+2 < self.pt_begin_layer:
                            counter += 1
                    # print("inside layer tune vis", counter)    
                else:
                    if (len(compound_prompts_deeper) > 0) and (not (counter > len(compound_prompts_deeper) - 1)):
                        # Appending the learnable tokens in different way
                        # x -> [77, NCLS, DIM]
                        # First remove the learnable tokens from previous layer
                        prefix = x[:1, :, :]
                        suffix = x[1 + self.compound_prompt_nctx:, :, :]
                        # Create/configure learnable tokens of this layer
                        textual_context = compound_prompts_deeper[counter]
                        if txtcls_feat is not None:
                            textual_context  = self.txtcls_adapter(textual_context.unsqueeze(1),
                            (txtcls_feat.unsqueeze(0).to(prefix.device), None)).squeeze(1)
                        textual_context = textual_context.expand(x.shape[-2], -1, -1).permute(1, 0, 2).half()
                        x = torch.cat([prefix, textual_context, suffix], dim=0)
                        # Once done, update the counter, so that the next time, it does not use same learnable tokens
                        counter += 1
                    if (txtcls_ctx_pt is not None):
                        if ((self.pt_begin_layer == 0) and (not (counter > len(txtcls_ctx_pt) - 1))):
                            if self.init_txtcls_pt is False: 
                                x = torch.cat([x[:-(len(txtcls_ctx_pt[counter]))], txtcls_ctx_pt[counter]], dim=0)
                            else:
                                x = torch.cat([x[:1], txtcls_ctx_pt[counter], x[(len(txtcls_ctx_pt[counter]))+1:]], dim=0)
                            # pdb.set_trace()
                            # x = torch.cat([x[:1], txtcls_ctx_pt[counter], x[3:]], dim=0)  ### TODO!!!! position modified
                            if len(compound_prompts_deeper) <= 0:
                                counter += 1
                        elif (self.pt_begin_layer > 0):
                            if (counter+2 >= self.pt_begin_layer and counter+2 <= len(txtcls_ctx_pt) + self.pt_begin_layer -1):
                                temp_index = counter-self.pt_begin_layer+2
                                if self.init_txtcls_pt is False: 
                                    x = torch.cat([x[:-(len(txtcls_ctx_pt[temp_index]))], txtcls_ctx_pt[temp_index]], dim=0)
                                else:
                                    x = torch.cat([x[:1], txtcls_ctx_pt[temp_index], x[(len(txtcls_ctx_pt[temp_index]))+1:]], dim=0)
                                if len(compound_prompts_deeper) <= 0:
                                    counter += 1                                
                            elif counter+2 < self.pt_begin_layer and len(compound_prompts_deeper) <= 0:
                                counter += 1
                        # print("inside layer tune text", counter, len(compound_prompts_deeper))
        elif len(compound_prompts_deeper) > 1 and self.pt_begin_layer > 0 and self.text_layer is False:
            compound_prompts_deeper = [x[-(len(compound_prompts_deeper[0])):]] + compound_prompts_deeper
            x = x[:-(len(compound_prompts_deeper[0]))]
        elif origin_ctx is not None and self.text_layer is True:
            origin_x = torch.cat([x[:-(origin_ctx.shape[1])], origin_ctx.permute(1,0,2)], dim=0)
        if origin_ctx is not None and self.text_layer is True:
            origin_x = origin_x + self.attention(self.ln_1(origin_x))
            origin_x = origin_x + self.mlp(self.ln_2(origin_x))  


        if self.adapter:
            adapt_x = self.adaptermlp(x, prior=prior, text_context = text_context)
            x = x + adapt_x
 
        ### not sure after the prompt tuning layers, whether it is correct or not. 
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))  

        # if not (x == x).any():
        #     print("visual x, output", x, counter)     
        #     pdb.set_trace()   

        if txtcls_feat is not None:
            return ([x, compound_prompts_deeper, counter, txtcls_feat, origin_x], prior, text_context)
        elif txtcls_ctx_pt is not None:
            return ([x, compound_prompts_deeper, counter, txtcls_ctx_pt, origin_x], prior, text_context)
        else:
            return ([x, compound_prompts_deeper, counter], prior, text_context)



class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 adapter: bool=False, adapter_layers: List=[i for i in range(24)], adapter_num_layers: int=1, multi_cross: bool = False,
                 prompts_needed=0, text_layer=False, design_details=None):
        super().__init__()
        self.width = width
        self.layers = layers
        if design_details is None:
            current_trainer = None
        else:
            current_trainer = design_details['trainer']
        
        # if attn_mask is not None:
        #     pdb.set_trace()
        if current_trainer == 'MaPLe':
            self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock_MAPLE(width, heads, attn_mask, adapter=((i in adapter_layers) and adapter), adapter_num_layers=adapter_num_layers, multi_cross = multi_cross,
             text_layer = text_layer, design_details = design_details, i = i)
            for i in range(layers)])
        elif current_trainer == 'fixed_clip':
            self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, adapter=((_ in adapter_layers) and adapter), adapter_num_layers=adapter_num_layers, multi_cross = multi_cross) for _ in range(layers)])


    def forward(self, x: torch.Tensor, prior=None, text_context = None):
        # return self.resblocks((x,prior))[0]
        return self.resblocks((x, prior, text_context))


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, use_adapter: bool=True, adapter_layers: List=[_ for _ in range(24)],
     adapter_num_layers: int=1, multi_cross: bool=False, design_details = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, adapter=use_adapter, adapter_layers=adapter_layers,
         adapter_num_layers=adapter_num_layers, multi_cross = multi_cross, design_details = design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.patch_size = patch_size
        
    
    def gauss_2d(self, std, size):
        imw, imh = size
        x1, y1 = torch.meshgrid(torch.arange(0, imw+1), torch.arange(0, imh+1))
        gauss_map = torch.exp(-0.5*(torch.pow(x1-int(imw/2), 2).to(std[0].device)/torch.pow(std[0], 2) + \
                                     torch.pow(y1-int(imh/2), 2).to(std[0].device)/torch.pow(std[1], 2)))
        try:
            gauss_map = gauss_map / torch.max(gauss_map)
        except:
            pdb.set_trace()
        return gauss_map
    
    def forward(self, x: torch.Tensor, prior=None, text_context = None, context = None):
        bs, c, h, w = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        if text_context is not None:
            x = self.transformer(x,prior, text_context)
        else:
            x = self.transformer(x,prior)
   
        x = x[0].permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
            if context is not None:
                x2 = self.ln_post(x2) @ self.proj
                return x[:,0,:], x[:,1:,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2),  x2[:,1:,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2)

        return x[:,0,:], x[:,1:,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            state_dict = {}
            for k in checkpoint.keys():
                if not k.startswith('visual') and not k.startswith('transf'):
                    print(k)

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.positional_embedding.shape != state_dict[new_k].shape:
                            pdb.set_trace()
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 16
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 14, 14, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            # H = W = self.input_resolution // 32
                            # spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            print(self.positional_embedding.shape , state_dict[new_k].shape,self.input_resolution)
                            assert self.positional_embedding.shape == state_dict[new_k].shape
            # pdb.set_trace()
            u, w = self.load_state_dict(state_dict, False)
            u = [k for k in u if not 'adaptermlp' in k and not 'extractor' in k]
            print(u, w, 'are misaligned params in CLIPResNet')



class VisionTransformer_MAPLE(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 use_adapter: bool=True, adapter_layers: List=[_ for _ in range(24)], adapter_num_layers: int=1, multi_cross: bool=False,
                 design_details = None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        self.VPT_shallow = True
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        # self.transformer = Transformer(width, layers, heads, adapter=use_adapter, adapter_layers=adapter_layers, adapter_num_layers=adapter_num_layers, multi_cross = multi_cross)
        self.prompt_till_layer_visual = 0
        self.transformer = Transformer(width, layers, heads, 
        adapter=use_adapter, adapter_layers=adapter_layers, adapter_num_layers=adapter_num_layers, multi_cross = multi_cross,
        prompts_needed=self.prompt_till_layer_visual, design_details=design_details)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.patch_size = patch_size
        self.compound_prompt_nctx = design_details['maple_length']

    
    def forward(self, x: torch.Tensor, prior=None, text_context = None, context = None,
                 shared_ctx = None, compound_deeper_prompts = None):
        bs, c, h, w = x.shape
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if self.VPT_shallow:
            # pdb.set_trace()
            if len(shared_ctx.shape) == 3:  
                visual_ctx = shared_ctx.permute(1,0,2).half()
            else:
                visual_ctx = shared_ctx.expand(x.shape[0], -1, -1).half()
            x = torch.cat([x, visual_ctx], dim=1)
        else:
            assert self.prompt_till_layer_visual == 0
        
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND 

        if text_context is not None:
            x = self.transformer([x, compound_deeper_prompts, 0],prior, text_context)
        else:
            x = self.transformer([x, compound_deeper_prompts, 0],prior)
        if isinstance(x[0], List):
            x = x[0][0].permute(1, 0, 2)  # LND -> NLD
        else:
            x = x[0].permute(1, 0, 2)
        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
      
        if self.proj is not None:
            x = x @ self.proj    
            if context is not None:
                x2 = self.ln_post(x2) @ self.proj
                return x[:,0,:], x[:,1:x.shape[1]-self.compound_prompt_nctx ,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2),  x2[:,1:,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2)

        if x.shape[1] == (h//self.patch_size) * (w//self.patch_size) + 1:
            return x[:,0,:], x[:,1:x.shape[1],:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2)
        else:
            return x[:,0,:], x[:,1:x.shape[1]-self.compound_prompt_nctx,:].view(bs,h//self.patch_size, w//self.patch_size, -1).permute(0, 3, 1, 2)


    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            state_dict = {}
            for k in checkpoint.keys():
                if not k.startswith('visual') and not k.startswith('transf'):
                    print(k)

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

                    if 'positional_embedding' in new_k:
                        if self.positional_embedding.shape != state_dict[new_k].shape:
                            pdb.set_trace()
                            print(f'Resize the pos_embed shape from {state_dict[new_k].shape} to {self.positional_embedding.shape}')
                            cls_pos = state_dict[new_k][0:1, :]
                            H = W = self.input_resolution // 16
                            spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 14, 14, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            # H = W = self.input_resolution // 32
                            # spatial_pos = F.interpolate(state_dict[new_k][1:,].reshape(1, 7, 7, cls_pos.shape[1]).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
                            spatial_pos = spatial_pos.reshape(cls_pos.shape[1], H*W).permute(1, 0)
                            positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                            state_dict[new_k] = positional_embedding
                            print(self.positional_embedding.shape , state_dict[new_k].shape,self.input_resolution)
                            assert self.positional_embedding.shape == state_dict[new_k].shape
            # pdb.set_trace()
            u, w = self.load_state_dict(state_dict, False)
            u = [k for k in u if not 'adaptermlp' in k and not 'extractor' in k]
            print(u, w, 'are misaligned params in CLIPResNet')


class CLIPTextContextEncoder(nn.Module):
    def __init__(self, context_length=13,
                 vocab_size=49408,
                 transformer_width=512,
                 transformer_heads=8,
                 transformer_layers=12,
                 embed_dim=1024,
                 out_dim=256,
                 pretrained=None, **kwargs):
        super().__init__()

        self.pretrained = pretrained

        self.context_length = context_length

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.embed_dim = embed_dim

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()
            state_dict = {}  
            for k in checkpoint.keys():
                if k.startswith('transformer.'):
                    state_dict[k] = checkpoint[k] 
                if k == 'positional_embedding' or k == 'text_projection' or k.startswith('token_embedding') or k.startswith('ln_final'):
                    if k == 'positional_embedding' and checkpoint[k].size(0) > self.context_length:
                        checkpoint[k] = checkpoint[k][:self.context_length]
                        print('positional_embedding is tuncated from 77 to', self.context_length)
                    state_dict[k] = checkpoint[k]
            
            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in text encoder')


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_context(self, text, context):
        pdb.set_trace()
        x_text = self.token_embedding(text)  # n_clas, n_text, C
        K, N1, C = x_text.shape
        B, N2, C = context.shape

        eos_indx = text.argmax(dim=-1) + N2
        eos_indx = eos_indx.reshape(1, K).expand(B, K).reshape(-1)

        x_text = x_text.reshape(1, K, N1, C).expand(B, K, N1, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)

        x = torch.cat([x_text[:,:,0:1], context, x_text[:, :, 1:]], dim=2).reshape(B*K, N1+N2, C)

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(B, K, self.embed_dim)
        return x
    '''
    def forward(self, text_verb, text_object, context):
        # pdb.set_trace()
        K, N1_v = text_verb.shape
        K, N1_o = text_object.shape
        B, N2, C = context.shape
        
        
        texts = torch.cat([text_verb, text_object],dim=-1)
        x_text = self.token_embedding(texts)
        eos_indx = text_object.argmax(dim=-1) + N1_v + N2

        x_text = x_text.reshape(1, K, N1_v+N1_o, C).expand(B, K, N1_v+N1_o, C)
        context = context.reshape(B, 1, N2, C).expand(B, K, N2, C)
        x = torch.cat([context[:,:,:4], x_text[:,:,0:N1_v], context[:,:,4:], x_text[:, :, N1_v:]], dim=2).reshape(B*K, N1_v+N1_o+N2, C)
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), eos_indx] @ self.text_projection
        x = x.reshape(B, K, self.embed_dim)
        return x
    '''
    def forward(self, text):
        
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x= self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward_prompts(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


        return x
        
class CLIP_ResNet(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 use_adapter=True
                 ):
        super().__init__()

        self.context_length = context_length
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_adapter=use_adapter
            )
        
        self.text_encoder = CLIPTextContextEncoder(context_length=context_length,
                                                    vocab_size=49408,
                                                    transformer_width=transformer_width,
                                                    transformer_heads=transformer_heads,
                                                    transformer_layers=transformer_layers,
                                                    embed_dim=embed_dim)
        self.initialize_parameters()

    def initialize_parameters(self):
        # nn.init.normal_(self.token_embedding.weight, std=0.02)
        # nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        x_global, x_local = self.visual(image.type(self.dtype))
        return x_global.float(), x_local.float()

    def encode_text(self, text, context):
        return self.text_encoder(text, context)


    def init_weights(self, pretrained=None):
        self.visual.init_weights(pretrained=pretrained)
        self.text_encoder.init_weights(pretrained=pretrained)

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 use_adapter=True,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length
        design_details = kwargs['design_details']
        trainer = kwargs['design_details']['trainer']
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            if trainer == "MaPLe":
                self.visual = VisionTransformer_MAPLE(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_adapter=use_adapter,
                adapter_layers=kwargs["adapter_layers"],
                adapter_num_layers=kwargs["adapter_num_layers"],
                multi_cross = kwargs['multi_cross'],
                design_details=design_details
            )
            else:
                self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                use_adapter=use_adapter,
                adapter_layers=kwargs["adapter_layers"],
                adapter_num_layers=kwargs["adapter_num_layers"],
                multi_cross = kwargs['multi_cross'],
                design_details=design_details
            )                
        
        if trainer == "MaPLe":
            prompt_till_layer_text = design_details['language_depth']
            self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            prompts_needed=prompt_till_layer_text,
            text_layer=True,
            design_details=design_details
            )
        else:
            self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            design_details=design_details
            )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        if len(x) > 1:
            x = x[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, use_adapter=True, adapter_pos='all', adapter_num_layers=1, multi_cross = False, design_details = None):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64

    # pdb.set_trace()
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))
    if adapter_pos == 'all':
        adapter_layers = [z for z in range(vision_layers)]
    elif adapter_pos == 'front':
        adapter_layers = [z for z in range(vision_layers // 2)]
    elif adapter_pos == 'end':
        adapter_layers = [z for z in range(vision_layers//2, vision_layers)]
    elif adapter_pos == 'last':
        adapter_layers = [z for z in range(vision_layers-1, vision_layers)]
    elif adapter_pos == 'random':
        adapter_layers = [random.randint(0, vision_layers-1) for z in range(vision_layers//2)] 
    print("CLIP image resolution", image_resolution)
    print("CLIP layer number", vision_layers)
    # pdb.set_trace()

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, use_adapter=use_adapter, 
        adapter_layers=adapter_layers, adapter_num_layers=adapter_num_layers, multi_cross = multi_cross, design_details = design_details
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print('[INFO] missing_keys:', [ k for k in missing_keys if 'adapter' not in k])
    print('[INFO] unexpected_keys:', unexpected_keys)
    return model

_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False, return_sot=True) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    # all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    
    # if return_sot:
    #     all_tokens = [[sot_token] + _tokenizer.encode(text) for text in texts]
    # else:
    #     all_tokens = [_tokenizer.encode(text) + [eot_token] for text in texts]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result
