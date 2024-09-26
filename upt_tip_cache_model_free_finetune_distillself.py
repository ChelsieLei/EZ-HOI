"""
Unary-pairwise transformer for human-object interaction detection

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""


from builtins import Exception
import os
import torch
import torch.distributed as dist


from torch import nn, Tensor
from typing import Optional, List
from torchvision.ops.boxes import batched_nms, box_iou

from ops import binary_focal_loss_with_logits

import sys
from hico_list import hico_verb_object_list,hico_verbs,hico_verbs_sentence,hico_verbs_sentence_2
from vcoco_list import vcoco_verbs_sentence
sys.path.append('detr')
from detr.models import build_model
from util import box_ops
from util.misc import nested_tensor_from_tensor_list
import pdb
import CLIP_models_adapter_prior2
import torchvision
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F
from transformer_module import TransformerDecoderLayer, TransformerSALayer
from CLIP.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import clip 
from ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
import pickle, random
from tqdm import tqdm
from hico_text_label import hico_unseen_index, MAP_AO_TO_HOI
import hico_text_label
from vcoco_text_label import vcoco_hoi_text_label, MAP_AO_TO_HOI_COCO, HOI_TO_AO_COCO
import cv2
from hico_text_label import HICO_INTERACTIONS, obj_to_name, HOI_TO_AO, HOI_IDX_TO_ACT_IDX
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math

import copy

record_act_pred = {}

class SupCtsLoss(nn.Module):

    def __init__(self, temperature=0.07, scale_by_temperature=False):
        super(SupCtsLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = F.normalize(features, p=2, dim=2)
        batch_size = features.shape[0]
        bank_size = features.shape[1]  

        if labels is not None and mask is not None: 
            raise ValueError('Cannot define both `labels` and `mask`') 
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(batch_size,-1, 1).to(device)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.permute(0,2,1)).float().to(device)
        else:
            mask = mask.float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.permute(0,2,1)),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

 
        ident = torch.eye(bank_size).unsqueeze(dim=2)
        for i in range(batch_size - 1):
            ident = torch.dstack((ident,torch.eye(bank_size).unsqueeze(dim=2)))
        logits_mask = torch.ones_like(mask).to(device) - ident.permute(2,0,1).to(device)  
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row  = torch.sum(positives_mask , axis=2)     
        denominator = torch.sum(
        exp_logits * negatives_mask, axis=2, keepdims=True) + torch.sum(
            exp_logits * positives_mask, axis=2, keepdims=True)  
        
        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        

        log_probs = torch.sum(
            log_probs*positives_mask , axis=2)/ num_positives_per_row

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        

        loss_m = loss.sum()/torch.nonzero(loss).shape[0]

        return loss_m, loss


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer_CA(nn.Module):
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

        
        tgt2, attentions = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)

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



class RelationNet(nn.Module):
    def __init__(self, feature_size=1536, embed_size=128):
        super(RelationNet, self).__init__()
        self.embe_size = embed_size
        self.fc1 = nn.Linear(feature_size, self.embe_size//2)
        self.fc2 = nn.Linear(feature_size, self.embe_size//2)
        self.g_mlp = nn.Sequential(
            nn.Linear(self.embe_size, self.embe_size // 2),
            nn.ReLU(),
            nn.Linear(self.embe_size // 2, self.embe_size // 2),
            nn.ReLU(),
        )
        self.fc3 = nn.Linear(self.embe_size // 2, 1)
        self.sigm = nn.Sigmoid()
        with torch.no_grad():
            nn.init.zeros_(self.fc3.weight)
            nn.init.zeros_(self.fc3.bias)
        
    def forward(self, feat1, feat2):
        feat1 = self.fc1(feat1)
        feat2 = self.fc1(feat2)
        feat1_ex = feat1.unsqueeze(1).repeat(1, feat2.shape[0], 1)
        feat2_ex = feat2.unsqueeze(0).repeat(feat1.shape[0], 1, 1)
        relation_pairs = torch.cat((feat1_ex, feat2_ex), 2)

        relation = self.g_mlp(relation_pairs)
        relation = self.fc3(relation)
        relation = self.sigm(relation)
        return relation.squeeze(2)

_tokenizer = _Tokenizer()
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

class Weight_Pred(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear1 = MLP(input_dim=input_dim, hidden_dim=512, output_dim=128, num_layers=2)
        self.drop1 = nn.Dropout()
        self.linear2 = MLP(input_dim=128, hidden_dim=32, output_dim=3, num_layers=2)
    
    def forward(self, x):
        x = self.drop1(self.linear1(x))
        x = self.linear2(x)
        return F.sigmoid(x)

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
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
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
        tempa = self.attention(self.ln_1(x))
        x = x + tempa[0]  
        x = x + self.mlp(self.ln_2(x))    
        return x

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, adapter: bool=False, adapter_layers: List=[i for i in range(24)], adapter_num_layers: int=1):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)[0]
        # return self.resblocks((x,prior))




class Adapter(nn.Module):
    def __init__(self,
                input_size,
                 dropout=0.1,
                 adapter_scalar="1.0",
                 adapter_num_layers=1,
                 mem_adpt_self = False, 
                 SA_only = False,
                 pt_tune = False,
                 prior_size = None,
                 down_size = 64
                 ):
        super().__init__()
        self.n_embd = input_size
        self.down_size = down_size
        self.scale = float(adapter_scalar)

        self.down_proj_mem = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj_mem = nn.Linear(self.down_size, self.n_embd)
        self.adapter_num_layers = adapter_num_layers
        if prior_size == None:
            self.down_proj_prior = MLP(input_size, 128, self.down_size, 3)
        else:
            self.down_proj_prior = MLP(prior_size, 128, self.down_size, 3)

        self.dropout = dropout
        with torch.no_grad():
            if pt_tune is False:
                nn.init.kaiming_uniform_(self.down_proj_mem.weight, a=math.sqrt(5))
                nn.init.zeros_(self.down_proj_mem.bias)
            nn.init.zeros_(self.up_proj_mem.weight)
            nn.init.zeros_(self.up_proj_mem.bias)
    
        if SA_only is False and pt_tune is False:
            instance_decoder_layer = TransformerDecoderLayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)
        elif SA_only is False and pt_tune is True:
            instance_decoder_layer = TransformerDecoderLayer_CA(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)            
        else:
            instance_decoder_layer = TransformerSALayer(self.down_size, 2, self.down_size*2,
                                                self.dropout, 'relu', False)            
        self.mhsa_layers = _get_clones(instance_decoder_layer, adapter_num_layers)
        self.mem_adpt_self = mem_adpt_self

    def forward(self, x, prior = None):
        # if torch.isnan(x).any():
        #     pass
        tempa = self.down_proj_mem(x)
        if prior is None or self.mem_adpt_self is True:
            # pdb.set_trace()
            context = tempa ## 18(#instance) x batchsize x 64
            mask = None
        else:
            prior, mask = prior
            context = self.down_proj_prior(prior).transpose(0,1)

            
        tempa = self.non_linear_func(tempa) ## 197 x batchsize x 64

        for z, layer in enumerate(self.mhsa_layers):
            # pdb.set_trace()
            tempa = layer(tempa, context, tgt_mask=None,
                        memory_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=mask,
                        pos=None, query_pos=None)
        
        up = self.up_proj_mem(tempa)
        output = (up * self.scale) + x
        return output



class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text, txtcls_feat = None, txtcls_pt_list = None, origin_ctx = None):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Pass as the list, as nn.sequential cannot process multiple arguments in the forward pass
        if txtcls_feat is not None:
            combined = [x, compound_prompts_deeper_text, 0, txtcls_feat, origin_ctx]
        elif txtcls_pt_list is not None:
            combined = [x, compound_prompts_deeper_text, 0, txtcls_pt_list, origin_ctx]
        else:
            combined = [x, compound_prompts_deeper_text, 0, origin_ctx]  # third argument is the counter which denotes depth of prompt
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
        if isinstance(x, List):
            if len(x) == 5:
                origin_x = x[4]
            else:
                origin_x = None
            x = x[0]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        if origin_x is not None:
            return x, origin_x
        return x


class MultiModalPromptLearner(nn.Module):
    def __init__(self, args, classnames, clip_model, object_class_to_target_class):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = args.N_CTX
        ctx_init = args.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution

        self.object_class_to_target_class = object_class_to_target_class
        self.compound_prompts_depth = args.tune_LY  # max=12, but will create 11 such shared prompts
        # self.fix_pre_pt = args.fix_pre_pt
        self.seperate_pre_pt = args.seperate_pre_pt

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = n_ctx
            prompt = clip.tokenize(ctx_init, context_length=77, truncate=True)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        if ctx_dim == 768:
            vis_dim = 1024
        elif ctx_dim == 512:
            vis_dim = 768
        # pdb.set_trace()
        self.proj = nn.Linear(ctx_dim, vis_dim)
        # self.proj.half()
        self.ctx = nn.Parameter(ctx_vectors)

        ## prompts for the image text decription
        self.img_descrip_prompt = args.img_descrip_prompt
        # self.txtcls_descrip = args.txtcls_descrip
        if self.img_descrip_prompt is True:
            #### for the VLM text description adapted prompts
            self.img_txt_adapter = Adapter(ctx_dim, down_size=args.emb_dim)
            
        self.txtcls_pt = args.txtcls_pt
        self.de_txtcls = args.de_txtcls
        self.img_clip_pt  = args.img_clip_pt 
        self.init_txtcls_pt = args.init_txtcls_pt
            
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers

        # Minimum can be 1, which defaults to shallow MaPLe
        if self.txtcls_pt is False and self.seperate_pre_pt is False:
            # compound prompts
            self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
            for single_para in self.compound_prompts_text:
                nn.init.normal_(single_para, std=0.02)
        elif self.txtcls_pt is True and self.seperate_pre_pt is True:
            # compound prompts
            self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, ctx_dim))
                                                      for _ in range(self.compound_prompts_depth - 1)])
            for single_para in self.compound_prompts_text:
                nn.init.normal_(single_para, std=0.02)
        else:
            self.compound_prompts_text = []
        # Also make corresponding projection layers, for each prompt
        self.deunify_pt = args.deunify_pt
        self.fix_txt_pt = args.fix_txt_pt
        if self.fix_txt_pt is True:
            self.deunify_pt = True
        if self.deunify_pt is True:
            self.vis_ctx_pt = []
            for index in range(self.compound_prompts_depth):
                self.vis_ctx_pt.append(nn.Parameter(torch.randn(n_ctx, vis_dim)).to(ctx_vectors.device))
        else:
            single_layer = nn.Linear(ctx_dim, vis_dim)
            self.compound_prompt_proj_vis = _get_clones(single_layer, self.compound_prompts_depth-1)

        if self.txtcls_pt is True:
            self.txtcls_pt_adapter = Adapter(ctx_dim, pt_tune = True, down_size=args.emb_dim)
            self.txtcls_ctx_pt = []
            for index in range(self.compound_prompts_depth):
                self.txtcls_ctx_pt.append(nn.Parameter(torch.randn(n_ctx, ctx_dim)))

        if self.img_clip_pt is True:
            self.img_clip_pt_adapter = Adapter(vis_dim, pt_tune = True, prior_size = ctx_dim, down_size=args.emb_dim)
            self.clip_img_file = args.clip_img_file

        classnames = [name[11:].replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.eval = args.eval
        self.without_unseen_name = args.without_unseen_name
        self.zs_type = args.zs_type
        if self.txtcls_pt is True and self.seperate_pre_pt is False and \
             self.init_txtcls_pt is False:
            if args.eval is True and args.without_unseen_name is True:
                self.register_buffer("token_prompts_eval", embedding[:, :, :]) 
            else:
                self.register_buffer("token_prompts", embedding[:, :, :]) 
        else:
            self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.unseen_pt_inj = args.unseen_pt_inj
        if self.unseen_pt_inj is True:
            self.unseen_pt_adapter = Adapter(ctx_dim, down_size = args.emb_dim)
        self.pt_begin_layer = args.pt_begin_layer

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, img_descrip_priors = None, txtcls_feat = None, 
        select_HOI_index = None, unseen_text_priors = None, filenames = None):
        ctx = self.ctx

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        if img_descrip_priors is not None:
            visual_deep_prompts = []
            for index, layer in enumerate(self.compound_prompt_proj_vis):
                ctx_imgdescrip_ = self.img_txt_adapter(self.compound_prompts_text[index].unsqueeze(1).repeat(1, img_descrip_priors[0].shape[0], 1), 
                                img_descrip_priors)
                visual_deep_prompts.append(layer(ctx_imgdescrip_))
            first_ly_vis_pt = self.img_txt_adapter(self.ctx.unsqueeze(1).repeat(1, img_descrip_priors[0].shape[0], 1), 
                                img_descrip_priors)
            first_ly_vis_pt = self.proj(first_ly_vis_pt)
        else:
            visual_deep_prompts = []
            ##### TODO add the image CLIP combination here
            if self.img_clip_pt is True:
                clip_img_list = []
                for fn in filenames:
                    clip_img = pickle.load(open(os.path.join(self.clip_img_file, fn.split(".")[0]+"_clip.pkl"),'rb'))
                    clip_img /= clip_img.norm(dim=-1, keepdim=True)
                    clip_img_list.append(clip_img)
                img_clip_prior = torch.stack(clip_img_list)

            if self.deunify_pt is False:
                if self.txtcls_pt is True:
                    for index, layer in enumerate(self.compound_prompt_proj_vis):
                        temp_vis_pt = layer(self.txtcls_ctx_pt[index+1].to(self.ctx.device))
                        visual_deep_prompts.append(temp_vis_pt)
                    first_ly_vis_pt = self.proj(self.txtcls_ctx_pt[0].to(self.ctx.device))            
                else:
                    for index, layer in enumerate(self.compound_prompt_proj_vis):
                        temp_vis_pt = layer(self.compound_prompts_text[index])
                        visual_deep_prompts.append(temp_vis_pt)
                    first_ly_vis_pt = self.proj(self.ctx)
            else:
                visual_deep_prompts = self.vis_ctx_pt[1:]
                first_ly_vis_pt = self.vis_ctx_pt[0].to(self.ctx.device)                     
        
        if self.img_clip_pt is True:
            for index, vis_pt_i in enumerate(visual_deep_prompts):
                # pdb.set_trace()
                visual_deep_prompts[index] = self.img_clip_pt_adapter(vis_pt_i.to(self.ctx.device).unsqueeze(1).repeat(1,len(img_clip_prior),1),
                                (img_clip_prior.to(self.ctx.device), None))
            first_ly_vis_pt = self.img_clip_pt_adapter(first_ly_vis_pt.to(self.ctx.device).unsqueeze(1).repeat(1,len(img_clip_prior),1),
                                (img_clip_prior.to(self.ctx.device), None))
        
        origin_ctx = None
        if self.txtcls_pt is True:
            txtcls_pt_list = []
            for index in range(len(self.txtcls_ctx_pt)):
                if self.de_txtcls is True:
                    temp_pt = self.txtcls_ctx_pt[index].to(self.ctx.device).unsqueeze(1).repeat(1,len(select_HOI_index),1)
                else:
                    temp_pt = self.txtcls_pt_adapter(self.txtcls_ctx_pt[index].to(self.ctx.device).unsqueeze(1).repeat(1,len(select_HOI_index),1),
                                     (txtcls_feat[select_HOI_index].to(self.ctx.device).unsqueeze(1).repeat(1, len(self.txtcls_ctx_pt[index]), 1), None))
                if unseen_text_priors is not None:
                    # unseen_num = sum(unseen_text_priors[2])
                    temp_pt = temp_pt.permute(1, 0, 2)
                    temp_ind_unseen = torch.tensor(unseen_text_priors[2])
                    temp_ind_simi_seen = torch.tensor(unseen_text_priors[3])

                    temp_prior = torch.cat((unseen_text_priors[0], temp_pt[temp_ind_simi_seen].clone().detach(), temp_pt[temp_ind_unseen]), dim=1)
                    temp_mask = torch.cat((unseen_text_priors[1], torch.zeros((len(temp_prior), self.n_ctx*2)).to(temp_prior.device)), dim=1)
                    temp_pt[temp_ind_unseen] = self.unseen_pt_adapter(temp_pt[temp_ind_unseen].permute(1, 0, 2), 
                                    (temp_prior,temp_mask)).permute(1, 0, 2)
                    
                    temp_pt = temp_pt.permute(1, 0, 2)
                txtcls_pt_list.append(temp_pt)
            
            
            if self.seperate_pre_pt is False and self.init_txtcls_pt is False:
                if self.eval is True and self.without_unseen_name is True:
                    prompts = self.token_prompts_eval
                else:
                    prompts = self.token_prompts
                origin_ctx = prompts[:, -(self.n_ctx):, :].clone().detach()
                if self.pt_begin_layer == 0:
                    prompts = torch.cat(
                    [
                        prompts[:, :-(self.n_ctx), :],
                        txtcls_pt_list[0].permute(1,0,2),  # (dim0, *, dim)
                    ],
                    dim=1,
                    )
                else:
                    return prompts, first_ly_vis_pt, self.compound_prompts_text, visual_deep_prompts, txtcls_pt_list, origin_ctx
            elif self.seperate_pre_pt is False and self.init_txtcls_pt is True: 
                prefix = self.token_prefix
                suffix = self.token_suffix
                prompts = self.construct_prompts(txtcls_pt_list[0].permute(1,0,2), prefix, suffix)
            elif self.seperate_pre_pt is True:
                prefix = self.token_prefix
                suffix = self.token_suffix
                prompts = self.construct_prompts(ctx, prefix, suffix)
            
            # if torch.isnan(prompts).any():        
            #     pdb.set_trace()
            
            return prompts, first_ly_vis_pt, self.compound_prompts_text, visual_deep_prompts, txtcls_pt_list[1:], origin_ctx
        else:
            prefix = self.token_prefix
            suffix = self.token_suffix
            prompts = self.construct_prompts(ctx, prefix, suffix)
            return prompts, first_ly_vis_pt, self.compound_prompts_text, visual_deep_prompts, origin_ctx   # pass here original, as for visual 768 is required


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model, object_class_to_target_class):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(args, classnames, clip_model, object_class_to_target_class)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.transformer = clip_model.transformer
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
       

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        x = self.transformer(x)
        if len(x) > 1:
            x = x[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, label=None):
        tokenized_prompts = self.tokenized_prompts
        logit_scale = self.logit_scale.exp()

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, origin_ctx = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits



class UPT(nn.Module):
    """
    Unary-pairwise transformer

    Parameters:
    -----------
    detector: nn.Module
        Object detector (DETR)
    postprocessor: nn.Module 
        Postprocessor for the object detector
    interaction_head: nn.Module
        Interaction head of the network
    human_idx: int
        Index of the human class
    num_classes: int
        Number of action/interaction classes
    alpha: float
        Hyper-parameter in the focal loss
    gamma: float
        Hyper-parameter in the focal loss
    box_score_thresh: float
        Threshold used to eliminate low-confidence objects
    fg_iou_thresh: float
        Threshold used to associate detections with ground truth
    min_instances: float
        Minimum number of instances (human or object) to sample
    max_instances: float
        Maximum number of instances (human or object) to sample
    """
    def __init__(self,
        args,
        detector: nn.Module,
        postprocessor: nn.Module,
        model: nn.Module,
        object_embedding: torch.tensor,
        human_idx: int, num_classes: int,
        alpha: float = 0.5, gamma: float = 2.0,
        box_score_thresh: float = 0.2, fg_iou_thresh: float = 0.5,
        min_instances: int = 3, max_instances: int = 15,
        object_class_to_target_class: List[list] = None,
        object_n_verb_to_interaction: List[list] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.detector = detector
        self.postprocessor = postprocessor
        self.clip_head = model
        self.object_embedding = object_embedding
        self.multi_cross = kwargs['multi_cross']
        self.img_descrip_prompt = args.img_descrip_prompt
        self.txtcls_descrip = args.txtcls_descrip
        self.txtcls_pt = args.txtcls_pt
        self.fix_mem = args.fix_mem
        self.img_clip_pt = args.img_clip_pt
        self.clip_test = args.clip_test
        self.hoicls_txt = kwargs['hoicls_txt']
        self.fixed_clip_enctxt = kwargs['fixed_clip_enctxt']
        self.visual_output_dim = model.image_encoder.output_dim

        self.object_n_verb_to_interaction = np.asarray(
                                object_n_verb_to_interaction, dtype=float
                            )
        self.use_gt_boxes = args.use_gt_boxes
        self.without_unseen_name = args.without_unseen_name
        self.zs_type = args.zs_type
        self.unseen_text_priors = kwargs['unseen_text_priors']

        self.human_idx = human_idx
        self.num_classes = num_classes

        self.alpha = alpha
        self.gamma = gamma

        self.box_score_thresh = box_score_thresh
        self.fg_iou_thresh = fg_iou_thresh

        self.min_instances = min_instances
        self.max_instances = max_instances
        self.object_class_to_target_class = object_class_to_target_class
        self.num_anno = kwargs["num_anno"]

        self.use_distill = args.use_distill
        self.use_consistloss = args.use_consistloss

        self.num_classes = num_classes
        self.use_multi_hot = args.use_multi_hot
        self.obj_affordance = args.obj_affordance


        num_shot = args.num_shot
        file1 = args.file1
        self.file1 = file1

        if args.zs:
            self.zs_type = args.zs_type
            self.filtered_hoi_idx = hico_unseen_index[self.zs_type]
        else:
            self.filtered_hoi_idx = []
            self.zs_type = None

        # self.unseen_verb_idxs = []
        self.label_choice = args.label_choice
        self.img_align = args.img_align
        self.txt_align = args.txt_align
        if args.img_align is True:
            self.mem_adapter = Adapter(self.visual_output_dim, mem_adpt_self = True, down_size = args.emb_dim)
        if args.txt_align is True:
            self.txtmem_adapter = Adapter(self.visual_output_dim, mem_adpt_self = True, down_size = args.emb_dim)
        self.select_HOI_index = kwargs['select_HOI_index']
        self.one_hots_HO, self.sample_lens_HO = self.load_cache_model(file1=file1, feature='hum_obj',num_classes=self.num_classes, num_shot=num_shot, filtered_hoi_idx = self.filtered_hoi_idx, 
                                                                                            use_multi_hot=self.use_multi_hot, label_choice=self.label_choice, num_anno=self.num_anno,
                                                                                            hoicls_txt = kwargs['hoicls_txt'], select_HOI_index = self.select_HOI_index, vlmtxt=args.vlmtxt, args = args)
        if args.without_unseen_name is True and len(torch.nonzero(self.sample_lens_HO==0))>0:
            excluded_hoi = torch.nonzero(self.sample_lens_HO==0).squeeze(1)
            self.sample_lens_HO[excluded_hoi] = 1E6

        self.one_hots_HO, self.sample_lens_HO = self.one_hots_HO.cuda().float(), self.sample_lens_HO.cuda().float()

        self.individual_norm = True
        self.logits_type = args.logits_type #
        
        self.consist = True
        self.evaluate_type = 'detr' # gt, detr
        
        self.use_type = 'crop'
        self.beta_cache = torch.tensor(10)
        self.alpha_cache = torch.tensor(1.0)
        if self.logits_type == 'HO':
            self.vis_fuse = nn.Sequential(
                nn.Linear(self.visual_output_dim * 2, self.visual_output_dim),
                nn.ReLU(),
                nn.Linear(self.visual_output_dim, self.visual_output_dim),
                nn.ReLU(),
            )
        elif self.logits_type == "HO+U":
            self.vis_fuse = nn.Sequential(
                nn.Linear(self.visual_output_dim * 3, self.visual_output_dim*2),
                nn.ReLU(),
                nn.Linear(self.visual_output_dim * 2, self.visual_output_dim),
                nn.ReLU(),
            )            
        if self.multi_cross is True:
            self.text_context_downproj = MLP(self.visual_output_dim, 128, 64, 3)
        self.prior_type = args.prior_type
        self.finetune_adapter = True
        if self.prior_type == 'cbe':
            self.priors_initial_dim = self.visual_output_dim+5
        elif self.prior_type == 'cb':
            self.priors_initial_dim = 5
        elif self.prior_type == 'ce':
            self.priors_initial_dim = self.visual_output_dim+1
        elif self.prior_type == 'be':
            self.priors_initial_dim = self.visual_output_dim+4
        elif self.prior_type == 'c':
            self.priors_initial_dim = 1
        elif self.prior_type == 'b':
            self.priors_initial_dim = 4
        elif self.prior_type == 'e':
            self.priors_initial_dim = self.visual_output_dim
        else:
            raise NotImplementedError

        self.use_weight_pred = args.use_weight_pred
        if self.finetune_adapter:
            
            if args.eval is True and args.without_unseen_name is True:
                self.label_HO_eval = nn.Parameter(self.one_hots_HO, requires_grad=False)
            else:
                self.label_HO = nn.Parameter(self.one_hots_HO, requires_grad=False)
            if not self.use_weight_pred:
                self.logit_scale_HO = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)) 


        if args.use_insadapter:
            if args.prior_method == 0:
                self.priors_downproj = MLP(self.priors_initial_dim, 128, 64, 3) # old 512+5   
            elif args.prior_method == 1:
                self.priors_downproj = MLP(self.priors_initial_dim * 2, 128, 64, 3) # old 512+5   
            elif args.prior_method == 2:
                self.learnable_prior = nn.Parameter(torch.empty(args.vis_prompt_num, 64))
                nn.init.xavier_normal_(self.learnable_prior)

        self.no_interaction_indexes = [9, 23, 30, 45, 53, 64, 75, 85, 91, 95, 106, 110, 128, 145, 159, 169, 173, 185, 193, 197, 207, 213, 223, 231, 234, 238, 242, 246, 251, 256, 263, 272, 282, 289, 294, 304, 312, 324, 329, 335, 341, 347, 351, 355, 362, 367, 375, 382, 388, 392, 396, 406, 413, 417, 428, 433, 437, 444, 448, 452, 462, 473, 482, 487, 501, 505, 515, 527, 532, 537, 545, 549, 557, 561, 566, 575, 583, 587, 594, 599]
        self.HOI_IDX_TO_OBJ_IDX = [
                4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14,
                14, 14, 14, 14, 14, 14, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 39,
                39, 39, 39, 39, 39, 39, 39, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 56, 56, 56, 56,
                56, 56, 57, 57, 57, 57, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 60, 60,
                60, 60, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 3,
                3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 58,
                58, 58, 58, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 6, 6, 6, 6, 6,
                6, 6, 6, 62, 62, 62, 62, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 24, 24,
                24, 24, 24, 24, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 34, 34, 34, 34, 34,
                34, 34, 34, 35, 35, 35, 21, 21, 21, 21, 59, 59, 59, 59, 13, 13, 13, 13, 73,
                73, 73, 73, 73, 45, 45, 45, 45, 45, 50, 50, 50, 50, 50, 50, 50, 55, 55, 55,
                55, 55, 55, 55, 55, 55, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 67, 67, 67,
                67, 67, 67, 67, 74, 74, 74, 74, 74, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41,
                54, 54, 54, 54, 54, 54, 54, 54, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20,
                20, 10, 10, 10, 10, 10, 42, 42, 42, 42, 42, 42, 29, 29, 29, 29, 29, 29, 23,
                23, 23, 23, 23, 23, 78, 78, 78, 78, 26, 26, 26, 26, 52, 52, 52, 52, 52, 52,
                52, 66, 66, 66, 66, 66, 33, 33, 33, 33, 33, 33, 33, 33, 43, 43, 43, 43, 43,
                43, 43, 63, 63, 63, 63, 63, 63, 68, 68, 68, 68, 64, 64, 64, 64, 49, 49, 49,
                49, 49, 49, 49, 49, 49, 49, 69, 69, 69, 69, 69, 69, 69, 12, 12, 12, 12, 53,
                53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 72, 72, 72, 72, 72, 65, 65, 65, 65,
                48, 48, 48, 48, 48, 48, 48, 76, 76, 76, 76, 71, 71, 71, 71, 36, 36, 36, 36,
                36, 36, 36, 36, 36, 36, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31,
                31, 31, 31, 31, 31, 31, 31, 44, 44, 44, 44, 44, 32, 32, 32, 32, 32, 32, 32,
                32, 32, 32, 32, 32, 32, 32, 11, 11, 11, 11, 28, 28, 28, 28, 28, 28, 28, 28,
                28, 28, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 37, 77, 77, 77, 77, 77,
                38, 38, 38, 38, 38, 27, 27, 27, 27, 27, 27, 27, 27, 70, 70, 70, 70, 61, 61,
                61, 61, 61, 61, 61, 61, 79, 79, 79, 79, 9, 9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7,
                7, 7, 25, 25, 25, 25, 25, 25, 25, 25, 75, 75, 75, 75, 40, 40, 40, 40, 40,
                40, 40, 22, 22, 22, 22, 22
            ]
        self.obj_to_no_interaction = torch.as_tensor([169, 23, 75, 159, 9, 64, 193, 575, 45, 566, 329, 505, 417, 246,
                                                        30,  85, 128, 145, 185, 106, 324, 238, 599, 347, 213, 583, 355, 545,
                                                        515, 341, 473, 482, 501, 375, 231, 234, 462, 527, 537,  53, 594, 304,
                                                        335, 382, 487, 256, 223, 207, 444, 406, 263, 282, 362, 428, 312, 272,
                                                        91,  95, 173, 242, 110, 557, 197, 388, 396, 437, 367, 289, 392, 413,
                                                        549, 452, 433, 251, 294, 587, 448, 532, 351, 561])

        self.epoch = 0
        # self.use_deformable_attn = args.use_deformable_attn
        self.COCO_CLASSES = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', \
                    'fire hydrant','N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',\
                    'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', \
                    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', \
                    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', \
                    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', \
                    'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', \
                    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self.reserve_indices = [idx for (idx, name) in enumerate(self.COCO_CLASSES) if name != 'N/A']
        self.reserve_indices = self.reserve_indices + [91]
        self.reserve_indices = torch.as_tensor(self.reserve_indices)
        self.dataset = args.dataset
        self.hyper_lambda = args.hyper_lambda
        self.pseudo_label = args.pseudo_label
        self.tpt = args.tpt
        self.featmap_dropout = nn.Dropout(0.2)
        self.feat_mask_type = args.feat_mask_type
        self.language_aware = args.LA 
        self.use_insadapter = args.use_insadapter
        self.prior_method = args.prior_method
        self.LA_weight = args.LA_weight
        self.box_proj = args.box_proj
        if self.box_proj:
            self.box_proj_mlp = MLP(8, 128, self.visual_output_dim, num_layers=3)
        if self.use_weight_pred:
            num_branch = len(self.logits_type.split('+'))
            self.weight_pred = Weight_Pred(input_dim=self.visual_output_dim*3, output_dim=num_branch)
        if self.obj_affordance:
            self.obj_affordance_query = nn.Parameter(torch.empty(1, self.visual_output_dim, dtype=self.clip_head.dtype))  # to be optimized
            self.obj_affordance_learner = nn.MultiheadAttention(embed_dim=512*1, num_heads=1, dropout=0.3, batch_first=True)
        # if self.dataset == 'swig':
        #     self.verb2interaction = torch.as_tensor(kwargs["verb2interaction"])
        self.use_mlp_proj = kwargs["use_mlp_proj"]
        if self.use_mlp_proj:
            self.mlp_proj = MLP(512, 512, 512, 3)
        self.vcoco = True if args.dataset == 'vcoco' else False
        if args.dataset == 'vcoco':
            self.vcoco_rela = RelationNet()
        self.fix_txt_pt = args.fix_txt_pt
        self.act_descriptor_feat_select = kwargs['act_descriptor_feat_select']
        if len(self.act_descriptor_feat_select) > 0:
            self.act_descriptor_attn = Adapter(self.visual_output_dim, prior_size = self.visual_output_dim, down_size=args.emb_dim)

        self.origin_ctx =args.origin_ctx
        if self.origin_ctx is True:
            self.SupCtsLoss = SupCtsLoss()


    def get_attention_feature(self, query_feat, human_feat, object_feat, ftype='patch'):  ## xxx
        
        device = torch.device('cuda')
        
        human_feat = human_feat.flatten(2).to(device)
        object_feat = object_feat.flatten(2).to(device)
        key_feat = torch.cat([human_feat,object_feat],dim=-1)

        query_feat = query_feat.flatten(2).transpose(1,2).to(device)
        
        global_feat = query_feat.mean(1)

        weight_matrix = torch.bmm(query_feat, key_feat)
        weight_matrix = weight_matrix.float().softmax(-1)
        weight_query_feat = torch.bmm(weight_matrix, key_feat.transpose(1, 2).float()).mean(1)
        query_feat = weight_query_feat.half()
        return query_feat.cpu()


    def load_cache_model(self,file1, feature='uni',num_classes=117, num_shot=10, filtered_hoi_idx=[], use_multi_hot=False, label_choice='random', 
                         num_anno=None,  hoicls_txt = None, select_HOI_index = None, vlmtxt = 'gpt', args = None):  ## √

        if args is not None and args.dataset =='vcoco':
            HOI_TO_AO_hoi = HOI_TO_AO_COCO
            numcls = 236
        else:
            HOI_TO_AO_hoi = HOI_TO_AO
            numcls = 600

        labels = torch.tensor([HOI_TO_AO_hoi[i][0] for i in range(numcls)])
        labels = labels[select_HOI_index]
        # pdb.set_trace()
        labels = torch.nn.functional.one_hot(labels, num_classes=num_classes)
        return labels, torch.sum(labels, dim=0)

    def get_clip_feature(self,image):  ## xxx
        x = self.clip_model.visual.conv1(image.type(self.clip_model.dtype))
        
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip_model.visual.positional_embedding.to(x.dtype)
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        
        local_feat = self.clip_model.visual.transformer.resblocks[:11]((x,None))[0]
        # x = self.clip_model.visual.transformer(x)
        # x = x.permute(1, 0, 2)  # LND -> NLD
        return local_feat
    
    def _reset_parameters(self):  ## xxx
        for p in self.context_aware.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.layer_norm.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_prior_scores(self,
        x: Tensor, y: Tensor, scores: Tensor, object_class: Tensor
    ) -> Tensor:  ### √
        
        prior_h = torch.zeros(len(x), self.num_classes, device=scores.device)
        prior_o = torch.zeros_like(prior_h)
        
        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else self.hyper_lambda
        s_h = scores[x].pow(p)
        s_o = scores[y].pow(p)
        if self.dataset == 'swig':
            prior_h = s_h.unsqueeze(-1).repeat(1, self.num_classes)
            prior_o = s_o.unsqueeze(-1).repeat(1, self.num_classes)
            return torch.stack([prior_h, prior_o])
        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping 
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
            for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_h[pair_idx, flat_target_idx] = s_h[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_h, prior_o])
    
    def conditional_mask(self, mask_shape: tuple, uni_mask_coor, instance_mask_coor,):
        '''
        :params
            mask_shape: e.g., (7,7)
            instance_mask_coor: [x1, y1, x2, y2]
        '''
        num = len(uni_mask_coor)
        tmp_mask1, tmp_mask2 = torch.zeros((num, *mask_shape)), torch.zeros((num, *mask_shape))
        instance_mask_coor[:, 0] = (instance_mask_coor[:, 0] - uni_mask_coor[:, 0]) / (uni_mask_coor[:, 2] - uni_mask_coor[:, 0]) * mask_shape[0]
        instance_mask_coor[:, 2] = (instance_mask_coor[:, 2] - uni_mask_coor[:, 0]) / (uni_mask_coor[:, 2] - uni_mask_coor[:, 0]) * mask_shape[0]
        instance_mask_coor[:, 1] = (instance_mask_coor[:, 1] - uni_mask_coor[:, 1]) / (uni_mask_coor[:, 3] - uni_mask_coor[:, 1]) * mask_shape[1]
        instance_mask_coor[:, 3] = (instance_mask_coor[:, 3] - uni_mask_coor[:, 1]) / (uni_mask_coor[:, 3] - uni_mask_coor[:, 1]) * mask_shape[1]
        instance_mask_coor = instance_mask_coor.int()
        for i in range(num):
            tmp_mask1[i, instance_mask_coor[i, 0] : instance_mask_coor[i, 2], :] = 1
            tmp_mask2[i, :, instance_mask_coor[i, 1]: instance_mask_coor[i, 3]] = 1
        intersection = tmp_mask1.logical_and(tmp_mask2)
        return intersection

    def compute_roi_embeddings(self, features: OrderedDict, image_size: Tensor, region_props: List[dict],
                                 flag = 0, feat_new = None, fix_mem = False, vcoco = False, 
                                 hoitxt_features = None, selected_txt_cls = None):
        device = features.device
        boxes_h_collated = []; boxes_o_collated = []
        prior_collated = []; object_class_collated = []
        # pairwise_tokens_collated = []
        attn_maps_collated = []
        all_logits = []

        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)

        gt_feats_collated = []
        pair_feats_collated = []
        gt_all_logits = []
        pair_logits = []
        pair_prior = []
        gt_labels = []
        for b_idx, props in enumerate(region_props):
            local_features = features[b_idx]
            boxes = props['boxes']
            scores = props['scores']
            labels = props['labels']

            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            # Permute human instances to the top
            if not torch.all(labels[:n_h]==self.human_idx):
                h_idx = torch.nonzero(is_human).squeeze(1)
                o_idx = torch.nonzero(is_human == 0).squeeze(1)
                perm = torch.cat([h_idx, o_idx])
                boxes = boxes[perm]; scores = scores[perm]
                labels = labels[perm]
            # Skip image when there are no valid human-object pairs
            if n_h == 0 or n <= 1:
                boxes_h_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                boxes_o_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                object_class_collated.append(torch.zeros(0, device=device, dtype=torch.int64))
                prior_collated.append(torch.zeros(2, 0, self.num_classes, device=device))
                continue

            # Get the pairwise indices
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device)
            )
            # Valid human-object pairs
            x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid human-object pairs")
            x = x.flatten(); y = y.flatten()
            
            if self.use_gt_boxes is True:
                x_keep = torch.tensor(list(range(int(len(boxes)/2)))).to("cuda").to(torch.long)
                y_keep = torch.tensor(list(range(int(len(boxes)/2), len(boxes))) ).to("cuda").to(torch.long)

            sub_boxes = boxes[x_keep]
            obj_boxes = boxes[y_keep]
            lt = torch.min(sub_boxes[..., :2], obj_boxes[..., :2]) # left point
            rb = torch.max(sub_boxes[..., 2:], obj_boxes[..., 2:]) # right point
            union_boxes = torch.cat([lt,rb],dim=-1)

            spatial_scale = 1 / (image_size[0,0]/local_features.shape[1])
            if  feat_new is not None:
                union_features = torchvision.ops.roi_align(feat_new[b_idx].unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            else:
                union_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[union_boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            single_features = torchvision.ops.roi_align(local_features.unsqueeze(0),[boxes],output_size=(7, 7),spatial_scale=spatial_scale,aligned=True)
            
            if self.feat_mask_type == 0:
                union_features = self.featmap_dropout(union_features).flatten(2).mean(-1)
                single_features = self.featmap_dropout(single_features).flatten(2).mean(-1)
            elif self.feat_mask_type == 1:
                union_features = union_features.flatten(2).mean(-1)
                single_features = single_features.flatten(2).mean(-1)
            # 
            
            human_features = single_features[x_keep]
            object_features = single_features[y_keep]
            if self.individual_norm: ## todo should use norm during finetuning?? 
                concat_feat_original = torch.cat([human_features,object_features, union_features],dim=-1)
                human_features = human_features / human_features.norm(dim=-1, keepdim=True)
                object_features = object_features / object_features.norm(dim=-1, keepdim=True)
                union_features = union_features / union_features.norm(dim=-1, keepdim=True)
            else:
                concat_feat = torch.cat([human_features,object_features, union_features],dim=-1) 
                concat_feat = concat_feat/concat_feat.norm(dim=-1,keepdim=True) 

            
            
            if self.logits_type == 'HO+U':
                vis_feat = self.vis_fuse(torch.cat([union_features, human_features, object_features], dim=-1))

            elif self.logits_type == 'HO':
                vis_feat = self.vis_fuse(torch.cat([human_features, object_features], dim=-1))
            elif self.logits_type == 'U':
                vis_feat = union_features
            if self.img_align is True:
                adapter_feat = (self.mem_adapter(vis_feat.unsqueeze(1))).squeeze(1)
            else:
                adapter_feat = vis_feat
            if self.txt_align is True:
                adapt_hoitxt_features = self.txtmem_adapter(hoitxt_features.unsqueeze(0)).squeeze(0)
            else:
                adapt_hoitxt_features = hoitxt_features
            
            
            if len(self.act_descriptor_feat_select) == 2 and len(self.act_descriptor_feat_select[0]) > 0:
                adapt_hoitxt_features = \
                    (self.act_descriptor_attn(adapt_hoitxt_features.unsqueeze(0),
                     (self.act_descriptor_feat_select[0][self.act_descriptor_feat_select[1]], None))).squeeze(0)
            
            phi_union_HO = adapter_feat @ adapt_hoitxt_features.T
            if self.training is False and self.without_unseen_name is True:
                logits_cache_HO = ((phi_union_HO @ self.label_HO_eval) / self.sample_lens_HO) / 2
            else:
                logits_cache_HO = ((phi_union_HO @ self.label_HO) / self.sample_lens_HO) / 2

            if self.use_weight_pred:
                logits_weights = self.weight_pred(torch.cat([human_features,object_features, union_features], dim=-1))
                logits = logits_cache_HO * logits_weights[:, 0:1]
            else:
                logits = logits_cache_HO * self.logit_scale_HO

            boxes_h_collated.append(x_keep)
            boxes_o_collated.append(y_keep)
            object_class_collated.append(labels[y_keep])
            prior_collated.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )
            all_logits.append(logits)
        glb_feat = local_features.flatten(1).mean(-1)

        if self.use_consistloss:
            pair_prior = torch.cat(pair_prior, dim=1).prod(0)
            x_, y_ = torch.nonzero(pair_prior).unbind(1)
            return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated, gt_all_logits, glb_feat
        else:
            return all_logits, prior_collated, boxes_h_collated, boxes_o_collated, object_class_collated, gt_feats_collated, pair_feats_collated, glb_feat
          
    def recover_boxes(self, boxes, size):  
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        h, w = size
        scale_fct = torch.stack([w, h, w, h])
        boxes = boxes * scale_fct
        return boxes

    def associate_with_ground_truth(self, boxes_h, boxes_o, targets): ## for training
        n = boxes_h.shape[0]
        labels = torch.zeros(n, self.num_classes, device=boxes_h.device)

        gt_bx_h = self.recover_boxes(targets['boxes_h'], targets['size'])
        gt_bx_o = self.recover_boxes(targets['boxes_o'], targets['size'])
        
        x, y = torch.nonzero(torch.min(
            box_iou(boxes_h, gt_bx_h),
            box_iou(boxes_o, gt_bx_o)
        ) >= self.fg_iou_thresh).unbind(1)
        # print("pair gt,",len(x),len(y))
        # IndexError: tensors used as indices must be long, byte or bool tensors
        if self.dataset == 'swig' and self.training:
            if len(y) > 0:
                tgthoi_y = torch.as_tensor([self.unique_hois[origin_hoi_idx.item()] for origin_hoi_idx in targets['hoi'][y]], device=boxes_h.device)
                labels[x, tgthoi_y] = 1
        elif self.num_classes == 117 or self.num_classes == 24 or self.num_classes == 407:
            labels[x, targets['labels'][y]] = 1  ## target['labels']: verb/action
        else:
            labels[x, targets['hoi'][y]] = 1
        # print("#(labels==1) = ", torch.sum(labels))
        return labels

    def compute_interaction_loss(self, boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats,
    reduction = 'sum'): ### loss
        ## bx, bo: indices of boxes
        labels = torch.cat([
            self.associate_with_ground_truth(bx[h], bx[o], target)
            for bx, h, o, target in zip(boxes, bh, bo, targets)
        ])
        ### TODO check how to obtain the GT labels for each H-O pairs
        # if self.pseudo_label and self.zs_type =='unseen_verb':
        #     W = (self.text_embedding[torch.as_tensor(self.seen_verb_idxs)] @ self.text_embedding[torch.as_tensor(self.unseen_verb_idxs)].T).to(labels.device)
        #     W = W.T
        #     W /= W.norm(dim=1, keepdim=True) ## 20 * 97
        #     labels[:, torch.as_tensor(self.unseen_verb_idxs).to(labels.device)] = labels[:, torch.as_tensor(self.seen_verb_idxs).to(labels.device)] @ W.T

        prior = torch.cat(prior, dim=1).prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        num_one_label = torch.sum(labels)
        logits = torch.cat(logits) 
        logits = logits[x, y]; prior = prior[x, y]; labels = labels[x, y]
        
        n_p = len(torch.nonzero(labels))

        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            # n_p_distll = torch.as_tensor([n_p_distll], device='cuda')
            dist.barrier() 
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

            # dist.all_reduce(n_p_distll)
            # n_p_distll = (n_p_distll / world_size).item()
            # n_p = (n_p.true_divide(world_size)).item()
        
        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction=reduction,
            alpha=self.alpha, gamma=self.gamma
            )
        if n_p == 0:
            return loss *n_p
        else:
            if self.use_distill:
                raise NotImplementedError
                # loss_feat = F.l1_loss(pair_feats, gt_feats,reduction='sum')/gt_feats.shape[1]
                loss_feat = torch.sum(3.0 - torch.diag(pair_feats @ gt_feats.t())) 
                return loss  / n_p + max((1-self.epoch * 0.05), 0) * loss_feat / n_p_distll
            else:
                return loss / n_p

    def prepare_region_proposals(self, results): ## √ detr extracts the human-object pairs
        region_props = []
        for res in results:
            sc, lb, bx = res.values()
            keep = batched_nms(bx, sc, lb, 0.5)
            sc = sc[keep].view(-1)
            lb = lb[keep].view(-1)
            bx = bx[keep].view(-1, 4)
            
            keep = torch.nonzero(sc >= self.box_score_thresh).squeeze(1)

            is_human = lb == self.human_idx
            hum = torch.nonzero(is_human).squeeze(1)
            obj = torch.nonzero(is_human == 0).squeeze(1)
            n_human = is_human[keep].sum(); n_object = len(keep) - n_human
            # Keep the number of human and object instances in a specified interval
            if n_human < self.min_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.min_instances]
                keep_h = hum[keep_h]
            elif n_human > self.max_instances:
                keep_h = sc[hum].argsort(descending=True)[:self.max_instances]
                keep_h = hum[keep_h]
            else:
                keep_h = torch.nonzero(is_human[keep]).squeeze(1)
                keep_h = keep[keep_h]

            if n_object < self.min_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.min_instances]
                keep_o = obj[keep_o]
            elif n_object > self.max_instances:
                keep_o = sc[obj].argsort(descending=True)[:self.max_instances]
                keep_o = obj[keep_o]
            else:
                keep_o = torch.nonzero(is_human[keep] == 0).squeeze(1)
                keep_o = keep[keep_o]

            keep = torch.cat([keep_h, keep_o])

            region_props.append(dict(
                boxes=bx[keep],
                scores=sc[keep],
                labels=lb[keep],
            ))

        return region_props

    def postprocessing(self, boxes, bh, bo, logits, prior, objects, image_sizes, flag = 0): ### √
        n = [len(b) for b in bh]
        logits = torch.cat(logits)
        logits = logits.split(n)

        detections = []
        for bx, h, o, lg, pr, obj, size,  in zip(
            boxes, bh, bo, logits, prior, objects, image_sizes,
        ):
            pr = pr.prod(0)
            x, y = torch.nonzero(pr).unbind(1)
            scores = torch.sigmoid(lg[x, y])
            # if flag == 1:
            #     pdb.set_trace()      
            detections.append(dict(
                boxes=bx, pairing=torch.stack([h[x], o[x]]),
                scores=scores* pr[x, y] , labels=y, 
                objects=obj[x], size=size
            ))       
        return detections


    def get_prior(self, region_props, image_size, prior_method): ##  for adapter module training
        
        max_feat = self.priors_initial_dim
        max_length = max(rep['boxes'].shape[0] for rep in region_props)
        mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
        priors = torch.zeros((len(region_props),max_length, max_feat), dtype=torch.float32, device=region_props[0]['boxes'].device)
        img_h, img_w = image_size.unbind(-1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        
        for b_idx, props in enumerate(region_props):
            boxes = props['boxes'] / scale_fct[b_idx][None,:]
            scores = props['scores']
            labels = props['labels']
            is_human = labels == self.human_idx
            n_h = torch.sum(is_human); n = len(boxes)
            if n_h == 0 or n <= 1:
                print(n_h,n)
                # sys.exit()
            
            object_embs = self.object_embedding[labels.to(self.object_embedding.device)]
            if self.obj_affordance:
                affordance_embs = self.get_obj_affordances(labels, region_props[0]['boxes'].device)
                object_embs = affordance_embs.squeeze(1)

            mask[b_idx,:n] = False
            
            if self.prior_type == 'cbe':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
                priors[b_idx,:n,5:self.visual_output_dim+5] = object_embs
                # priors[b_idx,:n,512+5:] = unary_tokens
            elif self.prior_type == 'cb':
                priors[b_idx,:n,:5] = torch.cat((scores.unsqueeze(-1),boxes),dim=-1)
            elif self.prior_type == 'ce':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
                priors[b_idx,:n,1:self.visual_output_dim+1] = object_embs
            elif self.prior_type == 'be':
                priors[b_idx,:n,:4] = boxes
                priors[b_idx,:n,4:self.visual_output_dim+4] = object_embs
            elif self.prior_type == 'c':
                priors[b_idx,:n,:1] = scores.unsqueeze(-1)
            elif self.prior_type == 'b':
                priors[b_idx,:n,:4] = boxes
            elif self.prior_type == 'e':
                priors[b_idx,:n,:self.visual_output_dim] = object_embs
            else:
                raise NotImplementedError

        if prior_method == 0:
            priors = self.priors_downproj(priors)
        elif prior_method == 1:
            pair_wise_priors = []
            for b_idx, props in enumerate(region_props):
                boxes = props['boxes'] / scale_fct[b_idx][None,:]
                scores = props['scores']
                labels = props['labels']
                is_human = labels == self.human_idx
                n_h = torch.sum(is_human); n = len(boxes)
                if n_h == 0 or n <= 1:
                    pair_wise_priors.append(torch.zeros(0, 0), )
                    print(n_h,n)
                    continue
                instance_wise_prior = priors[b_idx, :n]
                # Get the pairwise indices
                x, y = torch.meshgrid(
                    torch.arange(n, device=instance_wise_prior.device),
                    torch.arange(n, device=instance_wise_prior.device)
                )
                # Valid human-object pairs
                x_keep, y_keep = torch.nonzero(torch.logical_and(x != y, x < n_h)).unbind(1)
                if len(x_keep) == 0:
                    # Should never happen, just to be safe
                    raise ValueError("There are no valid human-object pairs")
                
                # extract single roi features
                sub_prior = instance_wise_prior[x_keep]
                obj_prior = instance_wise_prior[y_keep]
                
                pair_wise_priors.append(torch.cat((sub_prior, obj_prior), dim=-1))
            
            max_length = max(p.shape[0] for p in pair_wise_priors)
            mask = torch.ones((len(region_props),max_length),dtype=torch.bool,device=region_props[0]['boxes'].device)
            priors = torch.zeros((len(region_props),max_length, max_feat*2), dtype=torch.float32, device=region_props[0]['boxes'].device)
            for b_idx, props in enumerate(region_props):
                num_pair = pair_wise_priors[b_idx].shape[0]
                if num_pair > 0:
                    mask[b_idx, :num_pair] = False
                    priors[b_idx, :num_pair] = pair_wise_priors[b_idx]
            priors = self.priors_downproj(priors)   
        elif prior_method == 2:
            priors = self.learnable_prior.unsqueeze(0).repeat(len(region_props), 1, 1)
            mask = torch.zeros((priors.shape[0], priors.shape[1]), dtype=torch.bool,device=region_props[0]['boxes'].device)

        return (priors, mask)
    
    def prepare_target_hois(self, targets, device):
        unique_hois, cnt = {}, 0
        tgt_ids = []
        for t in targets:
            for hoi in t["hoi"]:
                hoi_id = hoi.item()
                if self.training:
                    # Only consider the texts within each mini-batch
                    if hoi_id not in unique_hois:
                        unique_hois[hoi_id] = cnt
                        cnt += 1
                    tgt_ids.append(unique_hois[hoi_id])
                else:
                    # Consider all hois in the dataset
                    tgt_ids.append(hoi_id)
        tgt_ids = torch.as_tensor(tgt_ids, dtype=torch.int64, device=device)
        return unique_hois

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (2, M) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `attn_maps`: list
                Attention weights in the cooperative and competitive layers
            `size`: torch.Tensor
                (2,) Image height and width
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        batch_size = len(images)
        images_orig = [im[0].float() for im in images]
        images_clip = [im[1] for im in images]

        device = images_clip[0].device
        image_sizes = torch.as_tensor([
            im.size()[-2:] for im in images_clip
        ], device=device)
        image_sizes_orig = torch.as_tensor([
            im.size()[-2:] for im in images_orig
            ], device=device)

        if isinstance(images_orig, (list, torch.Tensor)):
            images_orig = nested_tensor_from_tensor_list(images_orig)
        features, pos = self.detector.backbone(images_orig)
        src, mask = features[-1].decompose()
        # assert mask is not None2
        hs, detr_memory = self.detector.transformer(self.detector.input_proj(src), mask, self.detector.query_embed.weight, pos[-1])

        outputs_class = self.detector.class_embed(hs) # 6x8x100x81 or 6x8x100x92
        outputs_coord = self.detector.bbox_embed(hs).sigmoid() # 6x8x100x4 
        # pdb.set_trace()
        if outputs_class.shape[-1] == 92:
            outputs_class = outputs_class[:, :, :, self.reserve_indices]
            assert outputs_class.shape[-1] == 81, 'reserved shape NOT match 81'

        results = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        results = self.postprocessor(results, image_sizes)
        if self.use_gt_boxes is True:
            from util import box_ops
            temp_box = box_ops.box_cxcywh_to_xyxy(torch.cat((targets[0]['boxes_h'], targets[0]['boxes_o'])))
            temp_img_h, temp_img_w = image_sizes.unbind(1)
            scale_fct = torch.stack([temp_img_w, temp_img_h, temp_img_w, temp_img_h], dim=1)
            region_props = [{'boxes':  (temp_box* scale_fct[:, None, :]).squeeze(0), 
            'scores': torch.ones((len(targets[0]['hoi'])*2)).to("cuda")*0.9999,
             "labels": torch.cat((torch.zeros((len(targets[0]['hoi']))).to("cuda"), targets[0]['object'])).to(torch.long)}]

        else:
            region_props = self.prepare_region_proposals(results)
        if self.use_insadapter:
            priors = self.get_prior(region_props,image_sizes, self.prior_method) ## priors: (prior_feat, mask): (batch_size*14*64, batch_size*14)
        else: 
            priors = None
        images_clip = nested_tensor_from_tensor_list(images_clip)

        #### obtian VLM text description features
        if self.img_descrip_prompt:
            text_descrip_list = []
            for iidx, tgti in enumerate(targets):
                text_descrip =  tgti['text_descrip'].split(".")
                text_descrip = [txti for txti in text_descrip if len(txti) > 0]
                text_context_inputs = clip.tokenize(text_descrip)
                with torch.no_grad():
                    text_context = self.fixed_clip_enctxt(text_context_inputs.to(device))
                    text_context = text_context / text_context.norm(dim=-1, keepdim=True)
                text_descrip_list.append(text_context)
            # pdb.set_trace()
            max_length = max(rep.shape[0] for rep in text_descrip_list)
            mask = torch.ones((len(text_descrip_list),max_length),dtype=torch.bool,device=device)
            img_descrip_priors = torch.zeros((len(text_descrip_list),max_length, text_context.shape[-1]), dtype=torch.float32, device=device)
            for iidx, txt_descp in enumerate(text_descrip_list):
                mask[iidx,:len(txt_descp)] = False
                img_descrip_priors[iidx,:len(txt_descp)] = txt_descp
            img_descrip_priors = (img_descrip_priors, mask)
        else:
            img_descrip_priors = None

        #####
        if self.txtcls_descrip is True:
            txtcls_feat = self.hoicls_txt[self.select_HOI_index]
        else:
            txtcls_feat = None

        if self.img_clip_pt is True:
            filenames = [tgti['filename'] for tgti in targets]
        else:
            filenames = None

        ### MaPLe clip
        if self.fix_txt_pt is False and self.clip_test is False:
            tokenized_prompts = self.clip_head.tokenized_prompts
            if self.txtcls_pt is True:
                prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, txtcls_pt_list, origin_ctx = \
                    self.clip_head.prompt_learner(img_descrip_priors = img_descrip_priors, txtcls_feat = self.hoicls_txt, 
                    select_HOI_index = self.select_HOI_index, unseen_text_priors = self.unseen_text_priors, filenames = filenames)
            else:
                prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision, origin_ctx = \
                    self.clip_head.prompt_learner(img_descrip_priors = img_descrip_priors, unseen_text_priors = self.unseen_text_priors,
                    filenames = filenames)
                txtcls_pt_list = None
            hoitxt_features, origin_txt_features = self.clip_head.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text, txtcls_feat, txtcls_pt_list, origin_ctx)

        else:
            hoitxt_features = self.hoicls_txt[self.select_HOI_index].to(device)
            if self.fix_txt_pt is True:
                tokenized_prompts = self.clip_head.tokenized_prompts
                _, shared_ctx, _, deep_compound_prompts_vision,_ = \
                    self.clip_head.prompt_learner(img_descrip_priors = img_descrip_priors, unseen_text_priors = self.unseen_text_priors,
                    filenames = filenames)
            
        if self.clip_test is False:
            feat_global, feat_local = self.clip_head.image_encoder(images_clip.decompose()[0], priors,
                                        shared_ctx = shared_ctx, compound_deeper_prompts = deep_compound_prompts_vision)
        else:
            feat_global, feat_local = self.clip_head.image_encoder(images_clip.decompose()[0], priors)
        origin_ctx = origin_ctx[:,0,:]

        if torch.isnan(feat_local).any():
            print("different local", feat_local[0], feat_local[1], feat_local[2], feat_local[3])
            pdb.set_trace()

                
        logits, prior, bh, bo, objects, gt_feats, pair_feats, glb_feat = self.compute_roi_embeddings(feat_local, image_sizes, region_props,
                                                                    flag = 0, fix_mem=self.fix_mem, vcoco = self.vcoco, 
                                                                    hoitxt_features = hoitxt_features)  ## , selected_txt_cls = selected_txt_cls

        
        gt_all_logits = None
        boxes = [r['boxes'] for r in region_props] 

        if self.training:
            try:
                interaction_loss = self.compute_interaction_loss(boxes, bh, bo, logits, prior, targets, gt_feats, pair_feats)
            except:
                pdb.set_trace()
            if self.origin_ctx is True:
                ctx_data = torch.cat((hoitxt_features, origin_ctx), dim=0).unsqueeze(0)
                ctx_labels = torch.tensor(list(range(len(hoitxt_features)))).unsqueeze(0).repeat(2,1).reshape (1, -1)
                cts_loss_avg, cts_loss = self.SupCtsLoss(ctx_data, ctx_labels)
                cts_loss = cts_loss[:, int(cts_loss.shape[-1]/2)].sum()/torch.nonzero(cts_loss[:, int(cts_loss.shape[-1]/2)]).shape[0]
                interaction_loss = interaction_loss + cts_loss

            if self.fix_mem is False and self.vcoco is False:
                target_classes_label = torch.cat([targets[i]['verb'] for i in range(len(targets))], dim=0).type(torch.long).to(device)
                if self.txt_align is False:
                    vis_seen = hoitxt_features[target_classes_label]
                else:
                    vis_seen = self.txtmem_adapter(hoitxt_features.unsqueeze(0)).squeeze(0)[target_classes_label]

                visseen_similar = cal_similarity(vis_seen, vis_seen)
                # try:
                txtseen_similar = cal_similarity(self.hoicls_txt.to(device)[target_classes_label], 
                                                self.hoicls_txt.to(device)[target_classes_label])
                    
                relation_ho_cls_seen = kl_loss(visseen_similar, txtseen_similar)
                if self.zs_type is not None and self.without_unseen_name is False:
                    if self.txt_align is False:
                        vis_unseen = hoitxt_features[[j for j in self.filtered_hoi_idx if HOI_TO_AO[j][1] in [HOI_TO_AO[i.item()][1] for i in target_classes_label]]]
                    else:
                        vis_unseen = self.txtmem_adapter(hoitxt_features.unsqueeze(0)).squeeze(0)[[j for j in self.filtered_hoi_idx if HOI_TO_AO[j][1] in [HOI_TO_AO[i.item()][1] for i in target_classes_label]]]
                    if len(vis_unseen) == 0:
                        loss_dict = dict(
                        interaction_loss=interaction_loss,
                        sem_loss = (relation_ho_cls_seen)*150,
                        )
                    else:
                        visunseen_similar = cal_similarity(vis_unseen, vis_unseen)
                        txt_unseen = self.hoicls_txt.to(device)[[j for j in self.filtered_hoi_idx if HOI_TO_AO[j][1] in [HOI_TO_AO[i.item()][1] for i in target_classes_label]]]
                        txtunseen_similar = cal_similarity(txt_unseen, txt_unseen)
                        relation_ho_cls_unseen = kl_loss(visunseen_similar, txtunseen_similar)

                        visall_similar = cal_similarity(torch.cat((vis_seen, vis_unseen), dim=0),
                                                    torch.cat((vis_seen, vis_unseen), dim=0))
                        txtall_similar = cal_similarity(torch.cat((self.hoicls_txt.to(device)[target_classes_label], txt_unseen), dim=0),
                                                    torch.cat((self.hoicls_txt.to(device)[target_classes_label], txt_unseen), dim=0))
                        relation_ho_cls_all = kl_loss(visall_similar, txtall_similar)
                        
                        loss_dict = dict(
                            interaction_loss=interaction_loss,
                            sem_loss = (relation_ho_cls_seen+relation_ho_cls_unseen+relation_ho_cls_all )/3*150,
                        )
                else:
                    loss_dict = dict(
                        interaction_loss=interaction_loss,
                        sem_loss = (relation_ho_cls_seen)*150,
                    )
            else:
                loss_dict = dict(interaction_loss=interaction_loss)
            if torch.isnan(interaction_loss).any():
                print("targets", targets)
                print("loss", loss_dict)
                pdb.set_trace()
            return loss_dict
        if len(logits) == 0:
            print(targets)
            return None

        detections = self.postprocessing(boxes, bh, bo, logits, prior, objects, image_sizes, flag = 0)
        return detections

def random_color():
    rdn = random.randint(1, 1000)
    b = int(rdn * 997) % 255
    g = int(rdn * 4447) % 255
    r = int(rdn * 6563) % 255
    return b, g, r

def cal_similarity(key_embeds,
                   ref_embeds,
                   method='cosine',
                   temperature=-1):
    assert method in ['dot_product', 'cosine', 'euclidean']

    if key_embeds.size(0) == 0 or ref_embeds.size(0) == 0:
        return torch.zeros((key_embeds.size(0), ref_embeds.size(0)),
                           device=key_embeds.device)

    if method == 'cosine':
        key_embeds = F.normalize(key_embeds, p=2, dim=1)
        ref_embeds = F.normalize(ref_embeds, p=2, dim=1)
        return torch.mm(key_embeds, ref_embeds.t())
    elif method == 'euclidean':
        return euclidean_dist(key_embeds, ref_embeds)
    elif method == 'dot_product':
        if temperature > 0:
            dists = cal_similarity(key_embeds, ref_embeds, method='cosine')
            dists /= temperature
            return dists
        else:
            return torch.mm(key_embeds, ref_embeds.t())

def kl_loss(prediction, targets):
    T = 0.1
    return F.kl_div(F.log_softmax(prediction / T, dim=1),
             F.log_softmax(targets / T, dim=1),  # 1.2 0.1 0.2 0.3
             reduction='sum', log_target=True) / prediction.numel()


def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def get_multi_prompts(classnames):   ## https://github.com/openai/CLIP/blob/main/data/prompts.md, 
    templates = ['a photo of a person {}.',
                'a video of a person {}.',
                'a example of a person {}.',
                'a demonstration of a person {}.',
                'a photo of the person {}.',
                'a video of the person {}.',
                'a example of the person {}.', 
                'a demonstration of the person {}.',
                
                # 'a photo of a person during {}.',
                # 'a video of a person during {}.',
                # 'a example of a person during {}.',
                # 'a demonstration of a person during {}.',
                # 'a photo of the person during {}.',
                # 'a video of the person during {}.',
                # 'a example of the person during {}.',
                # 'a demonstration of the person during {}.',

                # 'a photo of a person performing {}.',
                # 'a video of a person performing {}.',
                # 'a example of a person performing {}.',
                # 'a demonstration of a person performing {}.',
                # 'a photo of the person performing {}.',
                # 'a video of the person performing {}.',
                # 'a example of the person performing {}.',
                # 'a demonstration of the person performing {}.',
                
                # 'a photo of a person practicing {}.',
                # 'a video of a person practicing {}.',
                # 'a example of a person practicing {}.',
                # 'a demonstration of a person practicing {}.',
                # 'a photo of the person practicing {}.',
                # 'a video of the person practicing {}.',
                # 'a example of the person practicing {}.',
                # 'a demonstration of the person practicing {}.',
                ]
    hico_texts = [' '.join(name.split(' ')[5:]) for name in classnames]
    all_texts_input = []
    for temp in templates:
        texts_input = torch.cat([clip.tokenize(temp.format(text)) for text in hico_texts ])
        all_texts_input.append(texts_input)
    all_texts_input = torch.stack(all_texts_input,dim=0)
    return all_texts_input

@torch.no_grad()
def get_origin_text_emb(args, clip_model, tgt_class_names, obj_class_names):
    use_templates = args.use_templates
    if use_templates == False:
        text_inputs = torch.cat([clip.tokenize(classname, context_length=77, truncate=True) for classname in tgt_class_names])
    elif use_templates:
        text_inputs = get_multi_prompts(tgt_class_names)
        bs_t, nums, c = text_inputs.shape
        text_inputs = text_inputs.view(-1, c)

    with torch.no_grad():
        text_inputs = text_inputs.to("cuda")
        origin_text_embedding = clip_model.encode_text(text_inputs)
    if use_templates:
        origin_text_embedding = origin_text_embedding.view(bs_t, nums, -1).mean(0)

    origin_text_embedding = origin_text_embedding / origin_text_embedding.norm(dim=-1, keepdim=True) # text embeddings of hoi 117*512 or 600*512

    obj_text_inputs = torch.cat([clip.tokenize(obj_text) for obj_text in obj_class_names]).to("cuda")
    with torch.no_grad():
        obj_text_embedding = clip_model.encode_text(obj_text_inputs)
        object_embedding = obj_text_embedding
        # obj_text_embedding = obj_text_embedding[hoi_obj_list,:]
    return origin_text_embedding, object_embedding


def build_detector(args, class_corr, object_n_verb_to_interaction, clip_model_path, num_anno, verb2interaction=None):
    if args.d_detr:
        detr, _, postprocessors = build_model_d_detr(args)
    else:
        detr, _, postprocessors = build_model(args)
    if os.path.exists(args.pretrained):
        if dist.get_rank() == 0:
            print(f"Load weights for the object detector from {args.pretrained}")
            # pdb.set_trace()
        if 'e632da11' in args.pretrained:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model']) 
        else:
            detr.load_state_dict(torch.load(args.pretrained, map_location='cpu')['model_state_dict'])
    
    clip_state_dict = torch.load(clip_model_path, map_location="cpu").state_dict()
    if args.clip_test is False:
        design_details = {"trainer": 'MaPLe',
                        "vision_depth": 0,
                        "language_depth": 0, "vision_ctx": 0,
                        "language_ctx": 0,
                        "maple_length": args.N_CTX,
                        "init_txtcls_pt": args.init_txtcls_pt,
                        "pt_begin_layer": args.pt_begin_layer
                        }
    else:
        design_details = {"trainer": "fixed_clip"}

    clip_model = CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=args.use_insadapter, adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers,
                         multi_cross = args.multi_cross, design_details = design_details)

    if args.num_classes == 117:
        classnames = hico_verbs_sentence
    elif args.num_classes == 24:
        classnames = vcoco_verbs_sentence
    elif args.num_classes == 600:
        classnames = list(hico_text_label.hico_text_label.values())
    else:
        raise NotImplementedError


    fixed_clip_model =  CLIP_models_adapter_prior2.build_model(state_dict=clip_state_dict, use_adapter=False,
    adapter_pos=args.adapter_pos, adapter_num_layers=args.adapter_num_layers, multi_cross = False, design_details = {"trainer": "fixed_clip"})
    fixed_clip_model.eval()
    fixed_clip_model = fixed_clip_model.to("cuda")

    obj_class_names = [obj[1] for obj in hico_text_label.hico_obj_text_label]


    if args.vlmtxt == 'llava':
        if args.dataset == 'hicodet':
            f2 = open("./hico_txt_llava/hico_HOI_descrip.txt","r")
            cls_descrip = []
            lines = f2.readlines()
            count = 0
            for line_i in lines:
                if count %3 == 1:
                    cls_descrip.append(list(hico_text_label.hico_text_label.values())[int(count/3)] +\
                                        ":" + line_i.split(":")[1][:-1])
                count += 1
            
        else:
            file_path = ("./hico_txt_llava/vcoco_HOI_descrip.txt")
            cls_descrip = []
            with open(file_path, 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]

            hico_txt_description = {}
            cur_key = 0
            for l_idx,line_i in enumerate(lines):
                if line_i[0] == '(' and line_i[1].isdigit():
                    act, obj = line_i.split(",")
                    act = int(act[1:])
                    obj = int(obj[1:-1])
                    hoii = MAP_AO_TO_HOI_COCO[(act, obj)]
                    cur_key = hoii
                    hico_txt_description[cur_key] = ''
                else:
                    hico_txt_description[cur_key] += line_i + ' '

            for  hoii in (hico_txt_description):
                tnt_dep = hico_txt_description[hoii][:300]
                quit_len = len(tnt_dep.split(".")[-1])
                if quit_len > 0:
                    tnt_dep = tnt_dep[:-quit_len]
                cls_descrip.append((vcoco_hoi_text_label)[HOI_TO_AO_COCO[int(hoii)]] +\
                                        ":" +tnt_dep)
        hoicls_txt, object_embedding = get_origin_text_emb(args, clip_model=fixed_clip_model, tgt_class_names=cls_descrip, obj_class_names=obj_class_names)
    else:
        hoicls_txt, object_embedding = get_origin_text_emb(args, clip_model=fixed_clip_model, tgt_class_names=list(hico_text_label.hico_text_label.values()), obj_class_names=obj_class_names)

    hoicls_txt = hoicls_txt.clone().detach().cpu()
    similarity_HOI = hoicls_txt @ hoicls_txt.T

    select_HOI_index = []
    for act_id in range(args.num_classes):
        if args.dataset == 'hicodet':
            if args.without_unseen_name is True and args.eval is False:
                bool_act_2_hoi = [i for i in range(len(HOI_IDX_TO_ACT_IDX)) if HOI_IDX_TO_ACT_IDX[i] == act_id and i not in hico_unseen_index[args.zs_type]]
            else:
                bool_act_2_hoi = [i for i in range(len(HOI_IDX_TO_ACT_IDX)) if HOI_IDX_TO_ACT_IDX[i] == act_id ]
        else:
            bool_act_2_hoi = [i for i in range(236) if HOI_TO_AO_COCO[i][0] == act_id ]
        temp_simi = similarity_HOI[bool_act_2_hoi]
        temp_simi = temp_simi.T[bool_act_2_hoi]
        if len(temp_simi) >2:
            ### select the most different HOI classes
            _, ind = torch.sort(temp_simi.view(-1))
            hoi_select1 = bool_act_2_hoi[int(ind[0]/len(temp_simi))]
            hoi_select2 = bool_act_2_hoi[ind[0]%len(temp_simi)]
            select_HOI_index.append(hoi_select1)
            select_HOI_index.append(hoi_select2)
        else:
            for act_i in bool_act_2_hoi:
                select_HOI_index.append(act_i)
    select_HOI_index.sort()
    # pdb.set_trace()
    
    if args.unseen_pt_inj is True:
        file_path = "hico_txt_llava/zs_possibility"
        if args.without_unseen_name is True:
            end_str = "_seen_diff.txt"
        else:
            end_str = "_diff.txt"
        cls_descrip = {}
        if args.dataset == 'hicodet':
            with open(os.path.join(file_path, args.zs_type + end_str), 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]
        else:
            with open(os.path.join(file_path, "vcoco_diff.txt"), 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]

        for line_idx in lines:
            if line_idx[0].isdigit():
                unseen_idx, similar_seen_idx = line_idx.split(",")
                unseen_idx = int(unseen_idx)
                # assert unseen_idx in hico_unseen_index[args.zs_type]
                similar_seen_idx = int(similar_seen_idx)
                cls_descrip[(unseen_idx, similar_seen_idx)] = ''
                current_pair = (unseen_idx, similar_seen_idx)
            else:
                cls_descrip[current_pair] = cls_descrip[current_pair] + line_idx
        if args.without_unseen_name is True and args.eval is True:
            with open(os.path.join(file_path, args.zs_type + "_diff.txt"), 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]
            for line_idx in lines:
                if line_idx[0].isdigit():
                    unseen_idx, similar_seen_idx = line_idx.split(",")
                    unseen_idx = int(unseen_idx)
                    # assert unseen_idx in hico_unseen_index[args.zs_type]
                    similar_seen_idx = int(similar_seen_idx)
                    cls_descrip[(unseen_idx, similar_seen_idx)] = ''
                    current_pair = (unseen_idx, similar_seen_idx)
                else:
                    cls_descrip[current_pair] = cls_descrip[current_pair] + line_idx
        if args.finetune_allcls_utpl is True and args.zs is False:
            with open(os.path.join(file_path, args.zs_type + "_seen_diff.txt"), 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]
            for line_idx in lines:
                if line_idx[0].isdigit():
                    unseen_idx, similar_seen_idx = line_idx.split(",")
                    unseen_idx = int(unseen_idx)
                    # assert unseen_idx in hico_unseen_index[args.zs_type]
                    similar_seen_idx = int(similar_seen_idx)
                    cls_descrip[(unseen_idx, similar_seen_idx)] = ''
                    current_pair = (unseen_idx, similar_seen_idx)
                else:
                    cls_descrip[current_pair] = cls_descrip[current_pair] + line_idx            


        text_descrip_list = {}
        for pairs_i in cls_descrip:
            if pairs_i[0] not in select_HOI_index:
                continue
            text_descrip =  cls_descrip[pairs_i].split(".")
            text_descrip = [txti for txti in text_descrip if len(txti) > 0]
            text_context_inputs = clip.tokenize(text_descrip)
            with torch.no_grad():
                text_context = fixed_clip_model.encode_text(text_context_inputs.to("cuda"))
            text_descrip_list[pairs_i] = text_context

        unseen_selected = [i[0] for i in text_descrip_list]
        seen_selected = [i[1] for i in text_descrip_list]
        select_HOI_index = list(set(select_HOI_index + seen_selected))
        select_HOI_index.sort()
        unseen_selected_id = [select_HOI_index.index(i) for i in unseen_selected]
        seen_selected_id = [select_HOI_index.index(i) for i in seen_selected]

        max_length = max(rep.shape[0] for rep in list(text_descrip_list.values()))
        mask = torch.ones((len(text_descrip_list),max_length),dtype=torch.bool,device="cuda")
        unseen_text_priors = torch.zeros((len(text_descrip_list),max_length, text_context.shape[-1]), dtype=torch.float32, device="cuda")
        related_similar_sen_idx = []
        for iidx, txt_descp in enumerate(text_descrip_list):
            mask[iidx,:len(text_descrip_list[txt_descp])] = False
            unseen_text_priors[iidx,:len(text_descrip_list[txt_descp])] = text_descrip_list[txt_descp]

        unseen_text_priors = (unseen_text_priors, mask, unseen_selected_id, seen_selected_id)
        if args.act_descriptor is True:
            act_descriptor_dict = {}
            with open(os.path.join(file_path, "all_act_descriptor.txt"), 'r') as file:
                lines = file.readlines()
                lines = [line.rstrip() for line in lines if line.rstrip()]
            for line_idx in lines:
                if (line_idx.split(",")[0]).isdigit() and ("," in line_idx and line_idx.split(",")[1][:9] == " a person"):
                    unseen_idx = line_idx.split(",")[0]
                    unseen_idx = int(unseen_idx)
                    act_descriptor_dict[unseen_idx] = []
                else:
                    if line_idx[0].isdigit() and line_idx[1:3] == ". ":
                        act_descriptor_dict[unseen_idx].append(line_idx)
            act_descriptor_feat_dict = {}
            sequence_selected = []
            for unseen_idx in act_descriptor_dict:
                if unseen_idx not in select_HOI_index:
                    continue

                text_context_inputs = clip.tokenize(act_descriptor_dict[unseen_idx],context_length=77, truncate=True)
                with torch.no_grad():
                    text_context = fixed_clip_model.encode_text(text_context_inputs.to("cuda"))
                act_descriptor_feat_dict[unseen_idx] = text_context
                sequence_selected.append(select_HOI_index.index(unseen_idx))
                if len(text_context)<3:
                    print("littel action descriptor, please regerenate descriptors for HOI class: ", unseen_idx)

            max_length = 3
            act_descriptor_feat_select = torch.zeros((len(act_descriptor_feat_dict),max_length, text_context.shape[-1]), dtype=torch.float32, device="cuda")
            for iidx, txt_descp in enumerate(act_descriptor_feat_dict):
                scr = act_descriptor_feat_dict[txt_descp].cpu() @ hoicls_txt[txt_descp]
                act_descriptor_feat_select[iidx] = act_descriptor_feat_dict[txt_descp][scr.topk(max_length)[1]]
            act_descriptor_feat_select = (act_descriptor_feat_select, sequence_selected)
          
        else:
            act_descriptor_feat_select = {}


    else:
        unseen_text_priors = None
        act_descriptor_feat_select = {}
    selected_classnames = [acti for idx, acti in  enumerate(list(hico_text_label.hico_text_label.values())) if idx in select_HOI_index]

    model = CustomCLIP(args, classnames=selected_classnames, clip_model=clip_model, object_class_to_target_class=class_corr)

    detector = UPT(args,
        detr, postprocessors['bbox'], model, object_embedding,
        human_idx=args.human_idx, num_classes=args.num_classes,
        alpha=args.alpha, gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        fg_iou_thresh=args.fg_iou_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        object_class_to_target_class=class_corr,
        object_n_verb_to_interaction=object_n_verb_to_interaction,
        num_anno = num_anno,
        # verb2interaction = verb2interaction,
        use_mlp_proj = args.use_mlp_proj,
        # cls_descrip = cls_descrip,
        hoicls_txt = hoicls_txt,
        multi_cross = args.multi_cross,
        select_HOI_index = select_HOI_index,
        fixed_clip_enctxt = fixed_clip_model.encode_text,
        unseen_text_priors = unseen_text_priors,
        act_descriptor_feat_select = act_descriptor_feat_select
    )
    return detector
