import warning
warning.filterwarnings('ignor')
from models.vit import VisionTransformer, interpolate_pos_embed 
from models.med import BertConfig, BertModel, BertLMHeadModel 
from transformers import BertTokenizer  #BERT tokenizer based on WordPiece 

import torch 
from torch import nn 
import torch.nn.functional as F

import os 
import urllib.parse import urlparse
from timm.models.hub import download_cached_file

def init_tokenizer():
    tokenizer =  BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token': '[DEC]'}) #add special tokens for encoder and decoder
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
    return tokenizer

def create_vit(vit, image_size, use_grad_checkpointing=False, ckpt_layer=0, drop_path_rate=0):
    assert vit in ['base', 'large'], 'vit parameters should be base or large'
    if vit=='base':
        vision_width = 768 
        visual_encoder = VisionTransformer(image_size=image_size, patch_size=16, embed_dim=vision_width, depth=12, num_heads=12,
                                           use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer, drop_path_rate=0 or drop_path_rate)
    elif vit=='large':
        vision_width = 1024 
        visual_encoder = VisionTransformer(image_size=image_size, patch_size=16, embed_dim=vision_width, depth=24, num_heads=16, 
                                           use_grad_checkpointing=use_grad_checkpointing, ckpt_layer=ckpt_layer, drop_path_rate=0.1 or drop_path_rate)
    
    return visual_encoder, vision_width

class BLIP_Decoder(nn.Module):
    def __init__(self, med_config='configs/med_config.json', image_size=384, vit='base', vit_grad_ckpt=False, vit_ckpt_layer=0, prompt='a picture of ',):
        super().__init__()
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)
        self.prompt = prompt 
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

    def forward(self, image, caption):
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id #'[CLS]'

        decoder_targets = text.input_ids.masked_fill(text.input_ids==self.tokenizer.pad_token_id, -100)
        decoder_targets[:, :self.prompt_length] = -100 #mask prompt and padding token when calculating loss
        
        decoder_output = self.text_decoder(text.input_ids, attention_mask=text.attention_mask, encoder_hidden_states=image_embeds,
                                           encoder_attention_mask=image_atts, labels=decoder_targets, return_dict=True,)
        loss_lm = decoder_output.loss

        return loss_lm

    def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds = self.visual_encoder(image)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0) #single image repeat three times to form a batch
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device) #attention_mask for image patch sequence
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
        prompt = [self.prompt] * image.size(0)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
        input_ids[:, 0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] #remove end token

        if sample: #nucleus sampling
            outputs = self.text_decoder.generate(input_ids=input_ids, max_length=max_length, min_length=min_length, do_sample=True, top_p=top_p,
                                                 num_return_sequences=1, eos_token_id=self.tokenizer.sep_token_id, pad_token_id=self.tokenizer.pad_token_id,
                                                 repetition_penalty=1.1, **model_kwargs)
        else: #beam search
            outputs = self.text_decoder.generate(input_ids=input_ids, max_length=max_length, min_length=min_length, num_beams=num_beams, eos_token_id=self.tokenizer.sep_token_id,
                                                  pad_token_id=self.tokenizer.pad_token_id, repetition_penalty=repetition_penalty, **model_kwargs)
        
        captions = []
        for output in outputs:
            caption = self.tokenizer.decode(output, skip_special_tokens=True)
            captions.append(caption[len(self.prompt):])
        return captions

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")
def load_checkpoint(model, url_or_filename):
    if is_url(url_or_filename):
        cached_file = download_cached_file(url_or_filename, check_hash=False, progress=True)
        checkpoint = torch.load(cached_file, map_location='cpu')
    elif os.path.isfile(url_or_filename):
        checkpoint = torch.load(url_or_filename, map_location='cpu')
    else:
        raise RuntimeError('checkpoint url or path is invalid')
    state_dict = checkpoint['model']
    state_dict['visual_encoder.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
    if 'visual_encoder_m.pos_embed' in model.state_dict().keys():
        state_dict['visual_encoder_m.pos_embed'] = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
    for key in model.state_dict().keys():
        if key in state_dict.keys():
            if state_dict[key].shape!=model.state_dict()[key].shape:
                del state_dict[key]
    msg = model.load_state_dict(state_dict, strict=False)
    print('load checkpoint from %s'%url_or_filename)  
    return model, msg          

def blip_decoder(pretrained='', **kwargs):
    model = BLIP_Decoder(**kwargs)
    if pretrained:
        model, msg = load_checkpoint(model, pretrained)
        assert(len(msg.missing_keys)==0)
    return model 








