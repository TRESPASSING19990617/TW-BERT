import copy

import numpy as np
import torch
import torch.nn.functional as F
from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
from einops import rearrange, reduce, repeat
import torch.distributed as dist
from horovod import torch as hvd
from src.modeling.cross_modal_bert.cross_models import *
from src.modeling.timesformer.vit_all import TimeSformer
from src.modeling.timesformer.vit_clip import TimeSformerCLIP
from src.modeling.xbert import (BertEmbeddings, BertEncoder, BertForMaskedLM,
                                BertLMPredictionHead, BertModel, BertPooler,
                                BertPreTrainedModel, BertPreTrainingHeads)
from src.utils.basic_utils import load_json, load_jsonl, save_frames_grid
from src.utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class TWBertBaseModel(nn.Module):
    def __init__(self, config=None, input_format='RGB', video_enc_cfg=None, cfg=None, temp=0.07):
        super().__init__()
        
        self.temp = nn.Parameter(torch.ones([]) * temp)   

        #config['num_hidden_layers']=6
        self.bert_config = config
        self.config = cfg

        visual_model_cls = eval(video_enc_cfg['cls'])
        video_enc_cfg['gradient_checkpointing']=True
        self.visual_encoder = visual_model_cls(model_cfg=video_enc_cfg, input_format=input_format, cross_attention_config=config)
        self.text_encoder = BertForMaskedLM.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.crossmodal_encoder = self._make_model()

        # FIXME make them configurable
        embed_dim = 256
        vision_width = 768
        self.embed_dim = embed_dim

        text_width = self.bert_config.hidden_size

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)         

        self.itc_token_type = self.bert_config.itc_token_type
        self.itm_head = nn.Linear(text_width*2, 2)
        
        self.queue_size = self.config['queue_size']
        self.queue_size_finegrain = 128
        self.momentum = self.config['momentum']  
        
        self.visual_encoder_m = visual_model_cls(model_cfg=video_enc_cfg, input_format=input_format, cross_attention_config=config)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained('bert-base-uncased', config=self.bert_config)
        self.text_proj_m = nn.Linear(text_width, embed_dim)    
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        
        self.copy_params()
        self.load_crossmodal_ckpt()

    def _make_model(self, N=6, d_model=768, d_ff=3072, h=12, dropout=0.1):

        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = Cross_MultiHeadedAttention(h, d_model, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        model = Cross_Encoder(Cross_EncoderLayer(d_model, c(attn), c(ff), dropout),
                              N, d_model)

        return model.cuda()
        
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)


    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

    @torch.no_grad()
    def load_crossmodal_ckpt(self):
        self.crossmodal_encoder.norm.bias.copy_(self.text_encoder.bert.encoder.layer[11].output.LayerNorm.bias)
        self.crossmodal_encoder.norm.weight.copy_(self.text_encoder.bert.encoder.layer[11].output.LayerNorm.weight)
        print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
        for i in range(6):
            self.crossmodal_encoder.layers[i].feed_forward.w_1.bias.copy_(self.text_encoder.bert.encoder.layer[i+6].intermediate.dense.bias)
            self.crossmodal_encoder.layers[i].feed_forward.w_1.weight.copy_(self.text_encoder.bert.encoder.layer[i+6].intermediate.dense.weight)
            self.crossmodal_encoder.layers[i].feed_forward.w_2.bias.copy_(self.text_encoder.bert.encoder.layer[i+6].output.dense.bias)
            self.crossmodal_encoder.layers[i].feed_forward.w_2.weight.copy_(self.text_encoder.bert.encoder.layer[i+6].output.dense.weight)

            self.crossmodal_encoder.layers[i].self_attn.linears[0].bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.query.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[0].weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.query.weight)
            self.crossmodal_encoder.layers[i].self_attn.linears[1].bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.key.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[1].weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.key.weight)
            self.crossmodal_encoder.layers[i].self_attn.linears[2].bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.value.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[2].weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.value.weight)

            self.crossmodal_encoder.layers[i].self_attn.linears[3].bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.query.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[3].weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.query.weight)
            self.crossmodal_encoder.layers[i].self_attn.linears[4].bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.key.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[4].weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.key.weight)
            self.crossmodal_encoder.layers[i].self_attn.linears[5].bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.value.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[5].weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.self.value.weight)

            self.crossmodal_encoder.layers[i].self_attn.linears[6].bias.copy_(self.text_encoder.bert.encoder.layer[i + 6].attention.output.dense.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[6].weight.copy_(self.text_encoder.bert.encoder.layer[i + 6].attention.output.dense.weight)
            self.crossmodal_encoder.layers[i].self_attn.linears[7].bias.copy_(self.text_encoder.bert.encoder.layer[i + 6].attention.output.dense.bias)
            self.crossmodal_encoder.layers[i].self_attn.linears[7].weight.copy_(self.text_encoder.bert.encoder.layer[i + 6].attention.output.dense.weight)

            #self.crossmodal_encoder.layers[i].self_attn.state_dict()["norm.weight"].copy_(self.text_encoder.bert.encoder.layer[i+6].attention.output.state_dict()["LayerNorm.weight"])
            self.crossmodal_encoder.layers[i].self_attn.norm.bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.output.LayerNorm.bias)
            self.crossmodal_encoder.layers[i].self_attn.norm.weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.output.LayerNorm.weight)
            #self.crossmodal_encoder.norm.weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.output.LayerNorm.weight)

            self.crossmodal_encoder.layers[i].sublayer[0].norm.bias.copy_(self.text_encoder.bert.encoder.layer[i+6].output.LayerNorm.bias)
            self.crossmodal_encoder.layers[i].sublayer[0].norm.weight.copy_(self.text_encoder.bert.encoder.layer[i+6].output.LayerNorm.weight)
            self.crossmodal_encoder.layers[i].sublayer[1].norm.bias.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.output.LayerNorm.bias)
            self.crossmodal_encoder.layers[i].sublayer[1].norm.weight.copy_(self.text_encoder.bert.encoder.layer[i+6].attention.output.LayerNorm.weight)
        for param in self.crossmodal_encoder.parameters():
            param.requires_grad = True


class TWBertForPretrain(TWBertBaseModel):
    def __init__(self, config, video_enc_cfg, cfg, input_format='RGB'):
        super(TWBertForPretrain, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg, cfg=cfg)

        self.use_mask_prob = 0
        
        self.register_buffer("video_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.video_queue = nn.functional.normalize(self.video_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)


    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']

        use_mpm = 'mpm_mask' in batch
        if use_mpm:
            context_visual_inputs = batch['context_visual_inputs']

        device = visual_inputs.device
        b, t, c, h, w = visual_inputs.shape

        text_embeds, text_feat = self._forward_text_feats(batch)
        # forward image and text features
        # feats are normalized embeds
        #if use_mpm and np.random.uniform() < self.use_mask_prob:
        #    video_embeds_total = self._forward_visual_embeds(torch.cat([visual_inputs, context_visual_inputs], dim=0))
            # split for unmasked and masked
        #    video_embeds, context_video_embeds = video_embeds_total[:b], video_embeds_total[b:]
        #else:
        video_embeds = self._forward_visual_embeds(visual_inputs, text_embeds[:, 0, :].unsqueeze(1))
        context_video_embeds = video_embeds

        # we compute normalized feats for unmasked visual inputs only, used for ITC
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)
        #video_feat_finegrain = F.normalize(self.vision_proj(video_embeds[:, 1:, :]), dim=-1)
        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)
        
        # text embeddings and features
        #text_embeds, text_feat = self._forward_text_feats(batch)
        #text_feat_finegrain = F.normalize(self.text_proj(text_embeds[:, :, :]), dim=-1)
        #print(text_feat.shape)

        # ========== (in-batch) ITC loss ==========
        if not self.training:
            gathered_video_feats = concat_all_gather(video_feat)
            gathered_text_feats = concat_all_gather(text_feat)
            #print(gathered_video_feats.shape, gathered_text_feats.shape)

            assert self.itc_token_type == 'cls', 'Support CLS tokens for ITC only, find {}.'.format(self.itc_token_type)
            sim_v2t = video_feat @ gathered_text_feats.t() / self.temp 
            sim_t2v = text_feat @ gathered_video_feats.t() / self.temp
            #print(sim_v2t.shape, sim_t2v.shape) 
                                
            # [IMPORTANT] be very careful when initializing the GT sim_v2t 
            # allgather return the concatenated features in the order of local_rank()
            sim_targets = torch.zeros_like(sim_v2t)

            local_rank = hvd.local_rank()
            b_start, b_end = b * local_rank, b * (local_rank + 1)
            sim_targets[:, b_start: b_end] = torch.eye(b)

            loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

            vtc_loss = (loss_v2t+loss_t2v) / 2

        # ========= (queue) ITC loss =========
        else:
            if 'alpha' in batch:
                alpha = batch['alpha']
            else:
                alpha = self.config['alpha']
                
            with torch.no_grad():
                self._momentum_update()
                
                text_output_m = self.text_encoder_m.bert(batch['text_input_ids'],
                                                attention_mask=batch['text_input_mask'],                      
                                                return_dict = True, 
                                                mode = 'text'
                                                )
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
                
                video_embeds_m = self.visual_encoder_m.forward_features(visual_inputs.transpose(1, 2), text_output_m.last_hidden_state[:,0,:].unsqueeze(1), return_all_tokens=True)
                video_feat_m = F.normalize(self.vision_proj_m(video_embeds_m[:,0,:]),dim=-1)
                video_feat_all = torch.cat([video_feat_m.t(), self.video_queue.clone().detach()], dim=1)

                #video_feat_m_finegrain = F.normalize(self.vision_proj_m(video_embeds_m[:, 1:, :]), dim=-1)
                #video_feat_all_finegrain = torch.cat([video_feat_m_finegrain, self.video_queue_finegrain.clone().detach()], dim=0)

                #text_feat_m_finegrain = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, :, :]), dim=-1)
                #text_feat_all_finegrain = torch.cat([text_feat_m_finegrain, self.text_queue_finegrain.clone().detach()],
                                                    #dim=0)

                #text_feat_m_mask_finegrain = batch['text_input_mask']
                #text_feat_all_mask_finegrain = torch.cat([text_feat_m_mask_finegrain, self.text_queue_mask_finegrain.clone().detach()], dim=0)

                #queue_size_finegrain = video_feat_all_finegrain.shape[0]
                
                sim_v2t_m = video_feat_m @ text_feat_all / self.temp 
                sim_t2v_m = text_feat_m @ video_feat_all / self.temp 
                #print(sim_v2t_m.shape)
                sim_targets = torch.zeros_like(sim_v2t_m)
                # local_rank = hvd.local_rank()
                # b_start, b_end = b * local_rank, b * (local_rank + 1)
                # sim_targets[:, b_start: b_end] = torch.eye(b)
                sim_targets.fill_diagonal_(1)
                
                sim_v2t_targets = alpha * F.softmax(sim_v2t_m, dim=1) + (1 - alpha) * sim_targets
                sim_t2v_targets = alpha * F.softmax(sim_t2v_m, dim=1) + (1 - alpha) * sim_targets    
            
            sim_v2t = video_feat @ text_feat_all / self.temp 
            sim_t2v = text_feat @ video_feat_all / self.temp 
                
            loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_v2t_targets,dim=1).mean()
            loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_t2v_targets,dim=1).mean()

            ####################################################################################################
            '''video_batch = video_feat_finegrain.shape[0]  # b
            text_batch = text_feat_finegrain.shape[0]  # b
            video_len = video_feat_finegrain.shape[1]  # 196
            text_len = text_feat_finegrain.shape[1]  # 40
            embed_dim = text_feat_finegrain.shape[2]  # 256

            text_feat_finegrain = text_feat_finegrain.view(-1, embed_dim)  # b*40 256
            video_feat_finegrain = video_feat_finegrain.view(-1, embed_dim)  # b*196 256
            text_feat_all_finegrain = text_feat_all_finegrain.view(-1, embed_dim)  # 128*40 256
            video_feat_all_finegrain = video_feat_all_finegrain.view(-1, embed_dim)  # 128*196 256

            sim_v2t_finegrain = video_feat_finegrain @ text_feat_all_finegrain.t() / self.temp  # b*196 128*40
            sim_t2v_finegrain = text_feat_finegrain @ video_feat_all_finegrain.t() / self.temp  # b*40 128*196

            sim_v2t_finegrain = sim_v2t_finegrain.view(video_batch, video_len, queue_size_finegrain,text_len).transpose(1, 2).transpose(2,3).contiguous()  # b 128 40 196
            sim_t2v_finegrain = sim_t2v_finegrain.view(text_batch, text_len, queue_size_finegrain, video_len).transpose(1, 2).contiguous()  # b 128 40 196

            sim_v2t_finegrain = sim_v2t_finegrain.view(video_batch, queue_size_finegrain, text_len, t, -1).max(-1)[0]  # b 128 40 t
            sim_t2v_finegrain = sim_t2v_finegrain.view(text_batch, queue_size_finegrain, text_len, t, -1).max(-1)[0]  # b 128 40 t

            sim_v2t_finegrain = sim_v2t_finegrain.masked_fill(text_feat_all_mask_finegrain.unsqueeze(0).unsqueeze(3) == 0, 0)  # b 128 40 t
            sim_t2v_finegrain = sim_t2v_finegrain.masked_fill(text_feat_m_mask_finegrain.unsqueeze(1).unsqueeze(3) == 0,0)  # b 128 40 t

            sim_v2t_mask = text_feat_all_mask_finegrain.sum(-1).view(1, queue_size_finegrain)  # 1 128
            sim_t2v_mask = text_feat_m_mask_finegrain.sum(-1).view(text_batch, 1)  # b 1

            sim_v2t_finegrain = sim_v2t_finegrain.view(video_batch, queue_size_finegrain, -1).sum(-1) / sim_v2t_mask / t  # b 128
            sim_t2v_finegrain = sim_t2v_finegrain.view(text_batch, queue_size_finegrain, -1).sum(-1) / sim_t2v_mask / t  # b 128

            sim_targets_finegrain = torch.zeros_like(sim_v2t_finegrain)
            sim_targets_finegrain.fill_diagonal_(1)
            # print(sim_v2t_finegrain.shape, sim_t2v_finegrain.shape)
            loss_v2t_finegrain = -torch.sum(F.log_softmax(sim_v2t_finegrain, dim=1) * sim_targets_finegrain,dim=1).mean()
            loss_t2v_finegrain = -torch.sum(F.log_softmax(sim_t2v_finegrain, dim=1) * sim_targets_finegrain,dim=1).mean()
            '''
            ####################################################################################################
            vtc_loss = (loss_v2t + loss_t2v) / 2
            
            self._dequeue_and_enqueue(video_feat_m, text_feat_m)

            #self._dequeue_and_enqueue_finegrain(video_feat_m_finegrain, text_feat_m_finegrain, text_feat_m_mask_finegrain)

        # ========= VTM ==========
        text_atts = batch['text_input_mask']

        # non-masked text and non-masked image 
        vtm_loss, vtm_logits, vtm_labels, encoder_outputs_pos = self.compute_vtm(text_embeds=text_embeds, 
                                                                                 text_atts=text_atts, 
                                                                                 video_embeds=video_embeds, 
                                                                                 video_atts=video_atts, 
                                                                                 sim_v2t=sim_v2t.clone(), # for hard mining
                                                                                 sim_t2v=sim_t2v.clone(), # for hard mining
                                                                                 T=int(t*0.65*0.65),
                                                                                 return_encoder_out=True
                                                                                )

        # ========= MLM ========== 
        # masked text and non-masked image
        if 'mlm_labels' in batch: 
            mlm_labels = batch['mlm_labels']
            mlm_text_input_ids = batch['mlm_text_input_ids']

            mlm_loss, mlm_logits, mlm_labels = self.compute_mlm(input_ids=mlm_text_input_ids,
                                                                text_input_mask=text_atts,
                                                                video_embeds=video_embeds, 
                                                                video_atts=video_atts,
                                                                mlm_labels=mlm_labels,
                                                                T=int(t*0.65*0.65)
                                                                )
        else:
            mlm_logits = mlm_loss = mlm_labels = None

        # ========= MPM ========== 
        if use_mpm: 
            mpm_labels, ignore_masks = self.get_pseudo_labels(batch)

            mpm_loss, mpm_logits = self.compute_mpm_with_encoder_out(encoder_outputs=encoder_outputs_pos, 
                                                                     text_atts=text_atts, 
                                                                     soft_labels=mpm_labels, 
                                                                     ignore_masks=ignore_masks, 
                                                                     patch_masks=batch['mpm_mask']
                                                                    )

        else:
            mpm_loss = mpm_logits = mpm_labels =  None

        return dict(
            itc_loss=vtc_loss,
            mlm_scores=mlm_logits,  # (B, Lt, vocab_size),  only text part
            mlm_loss=mlm_loss,  # (BxLt)
            mlm_labels=mlm_labels,  # (1, Lt), with -100 indicates ignored positions
            itm_scores=vtm_logits,  # (B, 2)
            itm_loss=vtm_loss,  # (1, )
            itm_labels=vtm_labels,  # (B, )
            mpm_loss=mpm_loss,
            mpm_logits=mpm_logits,
            mpm_labels=mpm_labels
        )


    def _forward_visual_embeds(self, visual_inputs, y):
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # image features
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, y, return_all_tokens=True)

        return video_embeds

    def _forward_text_feats(self, batch):
        # text features
        text_output = self.text_encoder.bert(batch['text_input_ids'], 
                                             attention_mask=batch['text_input_mask'],                      
                                             return_dict = True, 
                                             mode = 'text'
                                            )

        text_embeds = text_output.last_hidden_state # b, Lt, fsz=768
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)                 

        return text_embeds, text_feat

    def compute_mpm_with_encoder_out(self, encoder_outputs, text_atts, soft_labels, ignore_masks, patch_masks):
        txt_len = text_atts.shape[1]
        # adding one to ignore visual cls tokens
        '''visual_output = encoder_outputs.last_hidden_state[:, txt_len+1:]'''
        visual_output = encoder_outputs[:, txt_len + 1:]

        bsz, h, w = patch_masks.shape
        patch_masks_flatten_inverted = (1 - patch_masks.view(bsz, -1)).unsqueeze(-1)

        # mean embeds of masked visual regions
        num_masked_patches = torch.sum(patch_masks_flatten_inverted.squeeze(-1), dim=-1, keepdim=True)

        masked_visual_embeds = patch_masks_flatten_inverted * visual_output
        masked_visual_embeds = torch.sum(masked_visual_embeds, dim=1)
        masked_visual_embeds /= num_masked_patches

        # loss
        mpm_logits = self.mpm_head(masked_visual_embeds)

        cross_entropy = -torch.sum(F.log_softmax(mpm_logits, dim=1) * soft_labels, dim=1)
        cross_entropy[ignore_masks] = 0.

        mpm_loss = torch.sum(cross_entropy) / (bsz - torch.sum(ignore_masks))

        return mpm_loss, mpm_logits 

    def compute_mpm(self, text_embeds, text_atts, image_embeds, image_atts, soft_labels, ignore_masks, patch_masks, T):
        # forward cross-encoder
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs = self.crossmodal_encoder(embedding_output, attention_mask, T=T)
        '''encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )'''

        txt_len = text_atts.shape[1]
        # adding one to ignore visual cls tokens
        visual_output = encoder_outputs[:, txt_len + 1:]
        '''visual_output = encoder_outputs.last_hidden_state[:, txt_len+1:]'''

        bsz, h, w = patch_masks.shape
        patch_masks_flatten_inverted = (1 - patch_masks.view(bsz, -1)).unsqueeze(-1)

        # mean embeds of masked visual regions
        num_masked_patches = torch.sum(patch_masks_flatten_inverted.squeeze(-1), dim=-1, keepdim=True)

        masked_visual_embeds = patch_masks_flatten_inverted * visual_output
        masked_visual_embeds = torch.sum(masked_visual_embeds, dim=1)
        masked_visual_embeds /= num_masked_patches

        # loss
        mpm_logits = self.mpm_head(masked_visual_embeds)

        cross_entropy = -torch.sum(F.log_softmax(mpm_logits, dim=1) * soft_labels, dim=1)
        cross_entropy[ignore_masks] = 0.

        mpm_loss = torch.sum(cross_entropy) / (bsz - torch.sum(ignore_masks))

        return mpm_loss, mpm_logits 

    def compute_vtm(self, text_embeds, text_atts, video_embeds, video_atts, sim_v2t, sim_t2v, T, return_encoder_out=False):
        device = text_embeds.device

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, video_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs_pos = self.crossmodal_encoder(embedding_output_pos, attention_mask, T=T)
        '''encoder_outputs_pos = self.text_encoder.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )'''

        # ====== negative pairs =======
        bs = text_embeds.shape[0] 

        # local_rank = hvd.local_rank()
        # b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            # weights_i2t = sim_v2t[:,b_start:b_end]
            # weights_t2i = sim_t2v[:,b_start:b_end]
            weights_i2t = sim_v2t[:,:bs]
            weights_t2i = sim_t2v[:,:bs]
   
            # never select self as negative
            weights_i2t.fill_diagonal_(-1e10)
            weights_t2i.fill_diagonal_(-1e10)

            weights_i2t = F.softmax(weights_i2t, dim=1)
            weights_t2i = F.softmax(weights_t2i, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        video_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            video_embeds_neg.append(video_embeds[neg_idx])
        video_embeds_neg = torch.stack(video_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_atts, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([video_embeds_neg,video_embeds],dim=0)
        video_atts_all = torch.cat([video_atts,video_atts],dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.crossmodal_encoder(embedding_output_all, attention_mask_all, T=T)
        '''encoder_outputs_neg = self.text_encoder.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )'''

        '''vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                           encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)'''
        vl_embeddings1 = torch.cat([encoder_outputs_pos[:, 0, :],
                                   encoder_outputs_neg[:, 0, :]], dim=0)
        vl_embeddings2 = torch.cat([encoder_outputs_pos[:, text_atts.shape[1], :],
                                   encoder_outputs_neg[:, text_atts.shape[1], :]], dim=0)
        vl_embeddings = torch.cat([vl_embeddings1, vl_embeddings2], dim=1)
        #print(encoder_outputs_neg.shape)
        vtm_logits = self.itm_head(vl_embeddings)

        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)     

        if return_encoder_out:
            return vtm_loss, vtm_logits, vtm_labels, encoder_outputs_pos 
        else:
            return vtm_loss, vtm_logits, vtm_labels, None
        
    def compute_mlm(self, input_ids, text_input_mask, video_embeds, video_atts, mlm_labels, T):
        # forward text features with masked_input_ids
        text_output = self.text_encoder.bert(input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs = self.crossmodal_encoder(embedding_output, attention_mask, T=T)
        '''encoder_outputs = self.text_encoder.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )'''

        txt_len = text_input_mask.shape[1]
        txt_output = encoder_outputs[:, :txt_len]
        '''txt_output = encoder_outputs.last_hidden_state[:, :txt_len]'''

        mlm_logits = self.text_encoder.cls(txt_output)

        loss_fct = CrossEntropyLoss()
        mlm_loss = loss_fct(mlm_logits.view(-1, self.bert_config.vocab_size), mlm_labels.view(-1))

        return mlm_loss, mlm_logits, mlm_labels
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, video_feat, text_feat):
        # gather keys before updating queue
        video_feats = concat_all_gather(video_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = video_feats.shape[0]

        ptr = int(self.queue_ptr)
        #assert self.queue_size % batch_size == 0  # for simplicity

        if (ptr + batch_size) <= self.queue_size:
            self.video_queue[:, ptr:ptr + batch_size] = video_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        else:
            self.video_queue[:, ptr:self.queue_size] = video_feats.T[:,0:self.queue_size-ptr]
            self.text_queue[:, ptr:self.queue_size] = text_feats.T[:,0:self.queue_size-ptr]
            self.video_queue[:, 0:batch_size-self.queue_size + ptr] = video_feats.T[:,self.queue_size-ptr:]
            self.text_queue[:, 0:batch_size-self.queue_size + ptr] = text_feats.T[:,self.queue_size-ptr:]
        # replace the keys at ptr (dequeue and enqueue)
        #self.video_queue[:, ptr:ptr + batch_size] = video_feats.T
        #self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
        self.queue_ptr[0] = ptr

    
    def load_separate_ckpt(self, visual_weights_path=None, bert_weights_path=None, prompter_weights_path=None):
        if visual_weights_path:
            self.visual_encoder.load_state_dict(visual_weights_path)

        # [NOTE] BERT is initialized from huggingface pre-trained weights. 
        # if bert_weights_path:
        #     load_multimodal_encoder_state_dict_with_mismatch(self.cross_encoder, bert_weights_path)
        #     load_mlm_head_state_dict_with_mismatch(self.mlm_head, bert_weights_path)

        # TODO make path configurable
        #if prompter_weights_path is not None:
            #self.prompter.load_pretrained_weights_without_prompts(prompter_weights_path)


class TWBertForSequenceClassification(TWBertBaseModel):
    def __init__(self, config, video_enc_cfg, cfg, input_format='RGB'):
        super(TWBertForSequenceClassification, self).__init__(config, video_enc_cfg=video_enc_cfg, cfg=cfg)

        self.text_encoder = BertModel.from_pretrained('bert-base-uncased', config=self.bert_config, add_pooling_layer=False)      

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size,
                      config.hidden_size * 2),
            nn.ReLU(True),
            nn.Linear(config.hidden_size * 2, config.num_labels)
        )

    # def forward(self, image, text, targets, alpha=0, train=True):
    def forward(self, batch):
        visual_inputs = batch['visual_inputs']
        targets = batch['labels']

        device = visual_inputs.device

        # forward text
        text_input_mask = batch['text_input_mask']
        text_output = self.text_encoder(batch['text_input_ids'],
                                        attention_mask=text_input_mask,
                                        return_dict=True,
                                        mode='text'
                                        )
        text_embeds = text_output.last_hidden_state

        # forward visual
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)

        image_embeds, _ = self.visual_encoder.forward_features(visual_inputs, text_embeds[:, 0, :].unsqueeze(1), return_all_tokens=True)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        output = self.crossmodal_encoder(embedding_output, attention_mask, T=int(t*0.72*0.72))
        '''output = self.text_encoder(encoder_embeds=embedding_output,
                                attention_mask=attention_mask,
                                return_dict=True,
                                mode='fusion'
                                )'''

        prediction = self.classifier(torch.cat([output[:,0,:], output[:,text_input_mask.shape[1],:]], dim=1))
        #print(prediction.shape, targets.shape)
        if targets is not None:
            loss = F.cross_entropy(prediction, targets)                
        else: # evaluation mode
            loss = 0

        return dict(loss=loss,
                    logits=prediction
                    )
            

    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']
        device = visual_inputs.device

        # forward text
        text_input_mask = batch['text_input_mask']
        text_output = self.text_encoder.bert(batch['text_input_ids'],
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state

        # forward visual
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2)

        image_embeds = self.visual_encoder.forward_features(visual_inputs, text_embeds[:, 0, :].unsqueeze(1), return_all_tokens=True)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(device)

        # forward cross-encoder
        attention_mask = torch.cat([text_input_mask, image_atts], dim=1)
        embedding_output = torch.cat([text_embeds, image_embeds], dim=1)

        output = self.crossmodal_encoder(embedding_output, attention_mask, T=int(t*0.72*0.72))
        '''output = self.text_encoder.bert(encoder_embeds=embedding_output,
                                        attention_mask=attention_mask,
                                        return_dict=True,
                                        mode='fusion'
                                    )'''

        prediction = self.classifier(torch.cat([output[:,0,:], output[:,text_input_mask.shape[1],:]], dim=1))
        '''prediction = self.classifier(output.last_hidden_state[:,0,:])'''

        return prediction


class TWBertForVideoTextRetrieval(TWBertBaseModel):
    def __init__(self, config, video_enc_cfg, cfg, input_format='RGB'):
        super(TWBertForVideoTextRetrieval, self).__init__(config, input_format=input_format, video_enc_cfg=video_enc_cfg, cfg=cfg)
        
        self.register_buffer("video_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(self.embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.video_queue = nn.functional.normalize(self.video_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.copy_params()

    def forward(self, batch):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)

        visual_inputs = batch['visual_inputs']
        text_input_mask = batch['text_input_mask']
        text_input_ids = batch['text_input_ids']

        device = visual_inputs.device

        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        # visual embeddings
        
        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        
        visual_inputs = visual_inputs.transpose(1, 2)

        video_embeds = self.visual_encoder.forward_features(visual_inputs, text_embeds[:,0,:].unsqueeze(1), return_all_tokens=True)
        # image_embeds = image_embeds.repeat(text_input_mask.shape[0], 1, 1)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  

        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)

        # text embeddings

        # ========== (in-batch) ITC loss ==========
        if not self.training:
            gathered_video_feats = hvd.allgather(video_feat)
            gathered_text_feats = hvd.allgather(text_feat)

            sim_v2t = video_feat @ gathered_text_feats.t() / self.temp 
            sim_t2v = text_feat @ gathered_video_feats.t() / self.temp 

            sim_targets = torch.zeros_like(sim_v2t)

            local_rank = hvd.local_rank()
            b_start, b_end = b * local_rank, b * (local_rank + 1)
            sim_targets[:, b_start: b_end] = torch.eye(b)

            loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
            loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

            vtc_loss = (loss_v2t+loss_t2v) / 2

        # ========= (queue) ITC loss =========
        else:
            if 'alpha' in batch:
                alpha = batch['alpha']
            else:
                alpha = self.config['alpha']
                
            with torch.no_grad():
                self._momentum_update()
                text_output_m = self.text_encoder_m.bert(batch['text_input_ids'], 
                                                attention_mask=batch['text_input_mask'],                      
                                                return_dict = True, 
                                                mode = 'text'
                                                )
                text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1)
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)
                
                video_embeds_m = self.visual_encoder_m.forward_features(visual_inputs, text_output_m.last_hidden_state[:,0,:].unsqueeze(1), return_all_tokens=True)
                video_feat_m = F.normalize(self.vision_proj_m(video_embeds_m[:,0,:]),dim=-1)
                video_feat_all = torch.cat([video_feat_m.t(), self.video_queue.clone().detach()], dim=1)
                
                if self.config['distill']:
                
                    sim_v2t_m = video_feat_m @ text_feat_all / self.temp 
                    sim_t2v_m = text_feat_m @ video_feat_all / self.temp 

                    sim_targets = torch.zeros_like(sim_v2t_m)
    #                 # local_rank = hvd.local_rank()
    #                 # b_start, b_end = b * local_rank, b * (local_rank + 1)
    #                 # sim_targets[:, b_start: b_end] = torch.eye(b)
                    sim_targets.fill_diagonal_(1)

                    sim_v2t_targets = alpha * F.softmax(sim_v2t_m, dim=1) + (1 - alpha) * sim_targets
                    sim_t2v_targets = alpha * F.softmax(sim_t2v_m, dim=1) + (1 - alpha) * sim_targets 
            
            sim_v2t = video_feat @ text_feat_all / self.temp 
            sim_t2v = text_feat @ video_feat_all / self.temp 
            
            if self.config['distill']:
                loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_v2t_targets,dim=1).mean()
                loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_t2v_targets,dim=1).mean() 
            else:
                sim_targets = torch.zeros_like(sim_v2t)
                sim_targets.fill_diagonal_(1)
                loss_v2t = -torch.sum(F.log_softmax(sim_v2t, dim=1)*sim_targets,dim=1).mean()
                loss_t2v = -torch.sum(F.log_softmax(sim_t2v, dim=1)*sim_targets,dim=1).mean() 

            vtc_loss = [loss_v2t, loss_t2v]
            if False: # intra-modality alignment
                sim_v2v = video_feat @ video_feat_all / self.temp
                sim_t2t = text_feat @ text_feat_all / self.temp

                loss_v2v = -torch.sum(F.log_softmax(sim_v2v, dim=1)*sim_targets,dim=1).mean()
                loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets,dim=1).mean()
                vtc_loss.extend([loss_v2v, loss_t2t])

            vtc_loss = torch.stack(vtc_loss).mean()
            
            self._dequeue_and_enqueue(video_feat_m, text_feat_m)

        # ========= ITM ==========
        text_atts = batch['text_input_mask']

        # non-masked text and non-masked image 
        vtm_loss, vtm_logits, vtm_labels = self.compute_vtm(text_embeds=text_embeds, 
                                                            text_atts=text_atts, 
                                                            image_embeds=video_embeds, 
                                                            image_atts=video_atts, 
                                                            sim_i2t=sim_v2t.clone(), # for hard mining
                                                            sim_t2i=sim_t2v.clone(),  # for hard mining
                                                            T=int(t*0.65*0.65)
                                                           )

        return dict(
            itm_scores=vtm_logits,
            itm_loss=vtm_loss,
            itm_labels=vtm_labels,
            itc_loss=vtc_loss
        )
    
    def compute_vtm(self, text_embeds, text_atts, image_embeds, image_atts, sim_i2t, sim_t2i, T):
        device = text_embeds.device

        # ====== positive pairs =======
        attention_mask = torch.cat([text_atts, image_atts], dim=1)
        embedding_output_pos = torch.cat([text_embeds, image_embeds], dim=1)

        encoder_outputs_pos = self.crossmodal_encoder(embedding_output_pos, attention_mask, T=T)
        '''encoder_outputs_pos = self.text_encoder_fusion.bert(encoder_embeds=embedding_output_pos,
                                                     attention_mask=attention_mask,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )'''

        # ====== negative pairs =======
        bs = text_embeds.shape[0] 

        # local_rank = hvd.local_rank()
        # b_start, b_end = bs * local_rank, bs * (local_rank + 1)

        with torch.no_grad():
            # weights_v2t = sim_i2t[:,b_start:b_end]
            # weights_t2v = sim_t2i[:,b_start:b_end]
            weights_v2t = sim_i2t[:,:bs]
            weights_t2v = sim_t2i[:,:bs]
   
            # never select self as negative
            weights_v2t.fill_diagonal_(-np.Inf)
            weights_t2v.fill_diagonal_(-np.Inf)

            weights_v2t = F.softmax(weights_v2t, dim=1)
            weights_t2v = F.softmax(weights_t2v, dim=1)

        # select a negative image for each text
        # FIXME to optimize using indexing operations
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2v[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        # select a negative text for each image
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_v2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text_atts[neg_idx])

        text_embeds_neg = torch.stack(text_embeds_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg],dim=0)     
        text_atts_all = torch.cat([text_atts, text_atts_neg],dim=0)     

        video_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        video_atts_all = torch.cat([image_atts,image_atts],dim=0)

        attention_mask_all = torch.cat([text_atts_all, video_atts_all], dim=1)
        embedding_output_all = torch.cat([text_embeds_all, video_embeds_all], dim=1)

        # forward negative pairs via cross encoder
        encoder_outputs_neg = self.crossmodal_encoder(embedding_output_all, attention_mask_all, T=T)
        '''encoder_outputs_neg = self.text_encoder_fusion.bert(encoder_embeds=embedding_output_all,
                                                     attention_mask=attention_mask_all,
                                                     return_dict=True,
                                                     mode='fusion'
                                                    )'''

        #vl_embeddings = torch.cat([encoder_outputs_pos[:, 0, :],
                                   #encoder_outputs_neg[:, 0, :]], dim=0)
        vl_embeddings1 = torch.cat([encoder_outputs_pos[:, 0, :],
                                   encoder_outputs_neg[:, 0, :]], dim=0)
        vl_embeddings2 = torch.cat([encoder_outputs_pos[:, text_atts.shape[1], :],
                                   encoder_outputs_neg[:, text_atts.shape[1], :]], dim=0)
        vl_embeddings = torch.cat([vl_embeddings1, vl_embeddings2], dim=1)
        '''vl_embeddings = torch.cat([encoder_outputs_pos.last_hidden_state[:,0,:], 
                                   encoder_outputs_neg.last_hidden_state[:,0,:]],dim=0)'''
        vtm_logits = self.itm_head(vl_embeddings)            

        vtm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)], dim=0).to(device)
        vtm_loss = F.cross_entropy(vtm_logits, vtm_labels)     

        return vtm_loss, vtm_logits, vtm_labels 
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, video_feat, text_feat):
        # gather keys before updating queue
        video_feats = concat_all_gather(video_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = video_feats.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.queue_size % batch_size == 0  # for simplicity

        if (ptr + batch_size) <= self.queue_size:
            self.video_queue[:, ptr:ptr + batch_size] = video_feats.T
            self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        else:
            self.video_queue[:, ptr:self.queue_size] = video_feats.T[:, 0:self.queue_size - ptr]
            self.text_queue[:, ptr:self.queue_size] = text_feats.T[:, 0:self.queue_size - ptr]
            self.video_queue[:, 0:batch_size - self.queue_size + ptr] = video_feats.T[:, self.queue_size - ptr:]
            self.text_queue[:, 0:batch_size - self.queue_size + ptr] = text_feats.T[:, self.queue_size - ptr:]
        # replace the keys at ptr (dequeue and enqueue)
        # self.video_queue[:, ptr:ptr + batch_size] = video_feats.T
        # self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def forward_inference(self, batch):
        visual_inputs = batch['visual_inputs']
        text_input_mask = batch['text_input_mask']
        text_input_ids = batch['text_input_ids']
        #print(text_input_ids, text_input_mask)
        device = visual_inputs.device

        text_output = self.text_encoder.bert(text_input_ids,
                                             attention_mask=text_input_mask,
                                             return_dict=True,
                                             mode='text'
                                            )
        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)
        
        b, t, c, h, w = visual_inputs.shape
        # timeSformer asks for (b, c, t, h, w) as input.
        visual_inputs = visual_inputs.transpose(1, 2).repeat(text_input_mask.shape[0], 1, 1, 1, 1)
        video_embeds, scores= self.visual_encoder.forward_features(visual_inputs, text_embeds[:,0,:].unsqueeze(1), return_all_tokens=True)
        video_feat = F.normalize(self.vision_proj(video_embeds[:,0,:]),dim=-1)  #b m

        #video_embeds = video_embeds.repeat(text_input_mask.shape[0], 1, 1)
        # image_feat = image_feat.repeat(text_input_mask.shape[0], 1)
        video_atts = torch.ones(video_embeds.size()[:-1],dtype=torch.long).to(device)
                         

        #vtc_sim_scores = scores.unsqueeze(0) / self.temp # 1 b
        #print(vtc_sim_scores.shape, vtc_sim_scores)
        vtc_sim_scores = torch.diag(video_feat @ text_feat.t() / self.temp).unsqueeze(0)
        #print(vtc_sim_scores.shape, vtc_sim_scores)
        #vtc_sim_scores = torch.diag(vtc_sim_scores)

        attention_mask = torch.cat([text_input_mask, video_atts], dim=1)
        embedding_output = torch.cat([text_embeds, video_embeds], dim=1)

        encoder_outputs = self.crossmodal_encoder(embedding_output, attention_mask, T=int(t*0.65*0.65))
        '''encoder_outputs = self.text_encoder_fusion.bert(encoder_embeds=embedding_output,
                                                 attention_mask=attention_mask,
                                                 return_dict=True,
                                                 mode='fusion'
                                                )'''

        vl_embeddings1 = encoder_outputs[:, 0, :]
        vl_embeddings2 = encoder_outputs[:, text_input_mask.shape[1], :]
        vl_embeddings = torch.cat([vl_embeddings1, vl_embeddings2], dim=1)
        '''vl_embeddings = encoder_outputs.last_hidden_state[:,0,:]'''
        logits = self.itm_head(vl_embeddings)

        return dict(logits=logits, itc_scores=vtc_sim_scores)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    return hvd.allgather(tensor.contiguous())



