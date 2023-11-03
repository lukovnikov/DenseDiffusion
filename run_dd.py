import numpy as np
import torch
import requests 
import random
import os
import sys
import pickle
from PIL import Image

from tqdm.auto import tqdm
from datetime import datetime

import diffusers
from diffusers import DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn.functional as F

from utils import preprocess_mask, process_sketch, process_prompts, process_example

import fire
import json

from gligen.utils import seed_everything, display_bboxes, segimg_to_masks, _tokenize_annotated_prompt, mask_to_bbox
from drawlib import *
from pathlib import Path
import pickle as pkl


def mod_forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):

    residual = hidden_states

    if self.spatial_norm is not None:
        hidden_states = self.spatial_norm(hidden_states, temb)

    input_ndim = hidden_states.ndim

    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
    attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)

    if self.group_norm is not None:
        hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

    query = self.to_q(hidden_states)

    global sreg, creg, COUNT, creg_maps, sreg_maps, reg_sizes, text_cond
    
    sa_ = True if encoder_hidden_states is None else False
    encoder_hidden_states = text_cond if encoder_hidden_states is not None else hidden_states
        
    if self.norm_cross:
        encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

    key = self.to_k(encoder_hidden_states)
    value = self.to_v(encoder_hidden_states)

    query = self.head_to_batch_dim(query)
    key = self.head_to_batch_dim(key)
    value = self.head_to_batch_dim(value)
    
    if COUNT/32 < 50*0.3:
        
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()
            
        sim = torch.baddbmm(torch.empty(query.shape[0], query.shape[1], key.shape[1], 
                                        dtype=query.dtype, device=query.device),
                            query, key.transpose(-1, -2), beta=0, alpha=self.scale)
        
        treg = torch.pow(timesteps[COUNT//32]/1000, 5)
        
        ## reg at self-attn
        if sa_:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = sreg_maps[sim.size(1)].repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*sreg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*sreg*treg*(sim[int(sim.size(0)/2):]-min_value)
            
        ## reg at cross-attn
        else:
            min_value = sim[int(sim.size(0)/2):].min(-1)[0].unsqueeze(-1)
            max_value = sim[int(sim.size(0)/2):].max(-1)[0].unsqueeze(-1)  
            mask = creg_maps[sim.size(1)].repeat(self.heads,1,1)
            size_reg = reg_sizes[sim.size(1)].repeat(self.heads,1,1)
            
            sim[int(sim.size(0)/2):] += (mask>0)*size_reg*creg*treg*(max_value-sim[int(sim.size(0)/2):])
            sim[int(sim.size(0)/2):] -= ~(mask>0)*size_reg*creg*treg*(sim[int(sim.size(0)/2):]-min_value)

        attention_probs = sim.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)
            
    else:
        attention_probs = self.get_attention_scores(query, key, attention_mask)
           
    COUNT += 1
            
    hidden_states = torch.bmm(attention_probs, value)
    hidden_states = self.batch_to_head_dim(hidden_states)

    # linear proj
    hidden_states = self.to_out[0](hidden_states)
    # dropout
    hidden_states = self.to_out[1](hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if self.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / self.rescale_output_factor

    return hidden_states

      
#################################################
#################################################
def process_generation(binary_matrixes, seed, creg_, sreg_, sizereg_, bsz, master_prompt, *prompts):

    global creg, sreg, sizereg, device
    creg, sreg, sizereg = creg_, sreg_, sizereg_
    
    clipped_prompts = prompts[:len(binary_matrixes)]
    prompts = [master_prompt] + list(clipped_prompts)
    layouts = torch.cat([preprocess_mask(mask_, sp_sz, sp_sz, device) for mask_ in binary_matrixes])
    
    text_input = pipe.tokenizer(prompts, padding="max_length", return_length=True, return_overflowing_tokens=False, 
                                max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    cond_embeddings = pipe.text_encoder(text_input.input_ids.to(device))[0]

    uncond_input = pipe.tokenizer([""]*bsz, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                                  truncation=True, return_tensors="pt")
    uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]

    
    ###########################
    ###### prep for sreg ###### 
    ###########################
    global sreg_maps, reg_sizes
    sreg_maps = {}
    reg_sizes = {}
    
    for r in range(4):
        res = int(sp_sz/np.power(2,r))
        layouts_s = F.interpolate(layouts,(res, res),mode='nearest')
        layouts_s = (layouts_s.view(layouts_s.size(0),1,-1)*layouts_s.view(layouts_s.size(0),-1,1)).sum(0).unsqueeze(0).repeat(bsz,1,1)
        reg_sizes[np.power(res, 2)] = 1-sizereg*layouts_s.sum(-1, keepdim=True)/(np.power(res, 2))
        sreg_maps[np.power(res, 2)] = layouts_s


    ###########################
    ###### prep for creg ######
    ###########################
    pww_maps = torch.zeros(1,77,sp_sz,sp_sz).to(device)
    for i in range(1,len(prompts)):
        wlen = text_input['length'][i] - 2
        widx = text_input['input_ids'][i][1:1+wlen]
        for j in range(77):
            try:
                if (text_input['input_ids'][0][j:j+wlen] == widx).sum() == wlen:
                    pww_maps[:,j:j+wlen,:,:] = layouts[i-1:i]
                    cond_embeddings[0][j:j+wlen] = cond_embeddings[i][1:1+wlen]
                    break
            except:
                raise gr.Error("Please check whether every segment prompt is included in the full text !")
                return
    
    global creg_maps
    creg_maps = {}
    for r in range(4):
        res = int(sp_sz/np.power(2,r))
        layout_c = F.interpolate(pww_maps,(res,res),mode='nearest').view(1,77,-1).permute(0,2,1).repeat(bsz,1,1)
        creg_maps[np.power(res, 2)] = layout_c


    ###########################    
    #### prep for text_emb ####
    ###########################
    global text_cond
    text_cond = torch.cat([uncond_embeddings, cond_embeddings[:1].repeat(bsz,1,1)])    
    
    global COUNT
    COUNT = 0
    
    if seed == -1:
        latents = torch.randn(bsz,4,sp_sz,sp_sz).to(device)
    else:
        latents = torch.randn(bsz,4,sp_sz,sp_sz, generator=torch.Generator().manual_seed(seed)).to(device)
        
    image = pipe(prompts[:1]*bsz, latents=latents).images

    return(image)


def process_example_for_dd(example):
    assert len(example.captions) == 1
    caption = example.captions[0]
    promptpieces, regioncodes = _tokenize_annotated_prompt(caption)
    seg_img = example.load_seg_image()   #Image.open(self.panoptic_db[example_id]["segments_map"]).convert("RGB")
    seg_imgtensor = torch.tensor(np.array(seg_img)).permute(2, 0, 1)
    masks = segimg_to_masks(seg_imgtensor, regioncodes)
    # print(masks[0])
    binary_masks = []
    localprompts = []
    for mask, maskprompt in masks:
        binary_masks.append(mask)
        localprompts.append(maskprompt)
    return "".join(promptpieces), binary_masks, localprompts


def run_example(example, numgen=5, seed=-1, creg_=None, sreg_=None, sizereg_=None):
    caption, masks, prompts = process_example_for_dd(example)
    rets = []
    for i in range(numgen):
        gen_image = process_generation(masks, seed, creg_, sreg_, sizereg_, 1, caption, *prompts)[0]
        
        ret = {
            "master_prompt": caption,
            "masks": masks,
            "prompts": prompts,
            "seg_img": example.load_seg_image(),
            "image": gen_image,
        }
        rets.append(ret)
    return rets


def main(paths = [
            # "/USERSPACE/lukovdg1/controlnet11/evaldata/extradev.pkl",
            # "/USERSPACE/lukovdg1/controlnet11/evaldata/threeorange1.pkl",
            # "/USERSPACE/lukovdg1/controlnet11/evaldata/openair1.pkl",
            "/USERSPACE/lukovdg1/controlnet11/evaldata/threeballs1.pkl",
            # "/USERSPACE/lukovdg1/controlnet11/evaldata/threecoloredballs1.pkl",
            # "/USERSPACE/lukovdg1/controlnet11/evaldata/threesimplefruits1.pkl",
            # "/USERSPACE/lukovdg1/controlnet11/evaldata/foursquares2.pkl",
        ],
        #  [
        #     # "/USERSPACE/lukovdg1/controlnet11/evaldata/extradev.pkl",
        #     "/USERSPACE/lukovdg1/controlnet11/evaldata/threeorange1.pkl",
        #     # "/USERSPACE/lukovdg1/controlnet11/evaldata/threeballs1.pkl",
        #     # "/USERSPACE/lukovdg1/controlnet11/evaldata/threecoloredballs1.pkl",
        #     # "/USERSPACE/lukovdg1/controlnet11/evaldata/threesimplefruits1.pkl",
        #     # "/USERSPACE/lukovdg1/controlnet11/evaldata/openair1.pkl",
        #     # "/USERSPACE/lukovdg1/controlnet11/evaldata/foursquares2.pkl",
        # ],
        outpath="dd_outputs",
        seed=123456,
        gpu=0,
        am_c=1.,
        am_s=0.3,
        sizedeg=1.,
        numddimsteps=50,
    ):
    args = locals().copy()
    print(json.dumps(args, indent=4))
    
    ### check diffusers version
    if diffusers.__version__ != '0.20.2':
        print("Please use diffusers v0.20.2")
        sys.exit(0)

    global sreg, creg, sizereg, COUNT, reg_sizes, creg_maps, sreg_maps, pipe, text_cond
    global timesteps, sp_sz, device

    sreg = am_s
    creg = am_c
    sizereg = sizedeg
    COUNT = 0
    reg_sizes = {}
    creg_maps = {}
    sreg_maps = {}
    text_cond = 0
    device=torch.device("cuda", gpu)

    pipe = diffusers.StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            variant="fp16",
            # cache_dir='./models/diffusers/',
            # use_auth_token=HF_TOKEN
            safety_checker=None,
            ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.scheduler.set_timesteps(numddimsteps)
    timesteps = pipe.scheduler.timesteps
    sp_sz = pipe.unet.sample_size

    for _module in pipe.unet.modules():
        if _module.__class__.__name__ == "Attention":
            _module.__class__.__call__ = mod_forward
            

    outpath = Path(outpath)

    for path in [Path(p) for p in paths]:
        seed_everything(seed)
        
        i = 1
        print(f"Doing path: {path}")
        resultpath = None
        while resultpath is None or resultpath.exists():
            resultpath = outpath / (path.stem + f"_{i}_out")
            i += 1
            
        resultpath.mkdir(parents=True, exist_ok=True)
        print(f"Saving in : {resultpath}")

        with open(path, "rb") as f:
            inpexamples = pkl.load(f)

        allout = []
        for j, inpexample in enumerate(inpexamples):
            outputs = run_example(inpexample, seed=-1, creg_=creg, sreg_=sreg, sizereg_=sizereg)
            allout.append(outputs)
            imgs = [output["image"] for output in outputs]
            
            draw = DrawRow(*[DrawImage(img) for img in imgs])
            draw.drawself().save(resultpath / f"{j}.png")
            
        with open(resultpath / "outbatches.pkl", "wb") as outf:
            pkl.dump(allout, outf)


if __name__ == "__main__":
    fire.Fire(main)