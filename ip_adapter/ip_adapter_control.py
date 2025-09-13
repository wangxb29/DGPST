import os
from typing import List
import torch.nn as nn
import torch
from diffusers import StableDiffusionPipeline
from diffusers.pipelines.controlnet import MultiControlNetModel
from PIL import Image
from safetensors import safe_open
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import torch.nn.functional as F
from .utils import is_torch2_available, get_generator
if is_torch2_available():
    from .attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from .attention_processor_double import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
    from .attention_processor import (
        IPAttnProcessor2 as IPAttnProcessor2,
    )
    from .attention_processor_double import (
        SelfIPAttnProcessor2_0 as SelfIPAttnProcessor,
    )
else:
    from .attention_processor_double import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from .resampler import Resampler
from ip_adapter.utils import register_cross_attention_hook, get_net_attn_map, attnmaps2images

class LinearResampler(nn.Module):
    def __init__(
        self,
        input_dim=768,
        output_dim=768,
    ):
        super().__init__()
        self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projector(x)
    
class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens

        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class IPAdapter2(nn.Module):
    def __init__(self, sd_pipe, image_encoder_path, ip_ckpt, device, num_tokens=4):
        super().__init__()
        self.device = device
        self.image_encoder_path = image_encoder_path
        self.ip_ckpt = ip_ckpt
        self.num_tokens = 257
        self.pipe = sd_pipe.to(self.device)
        self.pipe.vae.requires_grad = False
        self.pipe.text_encoder.requires_grad = False
        for param in self.pipe.unet.parameters():
            param.requires_grad_(False)
        self.pipe.controlnet.requires_grad_(True)
        self.pipe.controlnet.train()
        self.pipe.unet = register_cross_attention_hook(self.pipe.unet)
        self.set_ip_adapter()
        self.adapter_modules = torch.nn.ModuleList(self.pipe.unet.attn_processors.values()).train()
        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float32
        ).eval()
        self.clip_image_processor = CLIPImageProcessor()
        # image proj model
        self.image_proj_model = self.init_proj3().train()
        self.image_proj_model2 = self.init_proj3().train()

        self.load_ip_adapter()
        
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=1024,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=torch.float32)
        return image_proj_model
    
    def init_proj2(self):
        image_proj_model = LinearResampler(
            input_dim=1280,
            output_dim=self.pipe.unet.config.cross_attention_dim,
        ).to(self.device, dtype=torch.float32)
        return image_proj_model
    
    def init_proj3(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.hidden_size,
        ).to(self.device, dtype=torch.float32)
        return image_proj_model
    
    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        unet_sd = unet.state_dict()
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if name.endswith("attn1.processor"):
                attn_procs[name] = AttnProcessor()
            else:
                scale = 1.0
                layer_name = name.split(".processor")[0]
                attn_procs[name] = SelfIPAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    scale=scale,
                    num_tokens=self.num_tokens,
                ).to(self.device, dtype=torch.float32)
                weights = {
                    "to_k_ips.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ips.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                    "to_q_ip.weight": unet_sd[layer_name + ".to_q.weight"],
                }    
                attn_procs[name].load_state_dict(weights,strict=False)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))#.to(self.device, dtype=torch.float32)

    def load_ip_adapter(self):
        if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("image_proj."):
                        state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                    elif key.startswith("ip_adapter."):
                        state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
        else:
            state_dict = torch.load(self.ip_ckpt, map_location="cpu")
        self.image_proj_model.load_state_dict(state_dict["image_proj"])
        self.image_proj_model2.load_state_dict(state_dict["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(state_dict["ip_adapter"], strict=False)
    
    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = mask.shape
        up = 2
        mask = mask.view(N, 1, 9, up, up, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(flow, [3,3], padding=1)
        up_flow = up_flow.view(N, -1, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, -1, up*H, up*W)    
    
    def get_image_embeds_corr(self, pil_image=None, clip_image_embeds=None, return_hidden=False):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

        clip_image = clip_image.to(self.device, dtype=torch.float32)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states
        clip_image_embeds = clip_image_embeds[-2]
        B, N, C = clip_image_embeds.size()
        # up_m = self.upmask(clip_image_embeds[:,1:,:].permute(0,2,1).reshape(B,C,16,16))
        # clip_image_embeds_up = self.upsample_flow(clip_image_embeds[:,1:,:].permute(0,2,1).reshape(B,C,16,16),up_m)
        # clip_image_embeds = torch.cat((clip_image_embeds[:,0:1,:],clip_image_embeds_up.reshape(B,C,-1).permute(0,2,1)),dim=1)

        image_prompt_embeds = self.image_proj_model(clip_image_embeds).to(dtype=torch.float32)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        # uncond_clip_image_embeds_up = self.upsample_flow(uncond_clip_image_embeds[:,1:,:].permute(0,2,1).reshape(B,C,16,16),up_m)
        # uncond_clip_image_embeds = torch.cat((uncond_clip_image_embeds[:,0:1,:],uncond_clip_image_embeds_up.reshape(B,C,-1).permute(0,2,1)),dim=1)
        uncond_image_prompt_embeds = self.image_proj_model(uncond_clip_image_embeds)
        if return_hidden:
            return image_prompt_embeds, uncond_image_prompt_embeds, clip_image_embeds
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def get_image_embeds_style(self, pil_image=None, clip_image_embeds=None, return_hidden=False):
        if isinstance(pil_image, Image.Image):
            pil_image = [pil_image]
        clip_image = self.clip_image_processor(images=pil_image, return_tensors="pt").pixel_values

        clip_image = clip_image.to(self.device, dtype=torch.float32)
        clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states
        clip_image_embeds = clip_image_embeds[-2]
        B, N, C = clip_image_embeds.size()
        # up_m = self.upmask(clip_image_embeds[:,1:,:].permute(0,2,1).reshape(B,C,16,16))
        # clip_image_embeds_up = self.upsample_flow(clip_image_embeds[:,1:,:].permute(0,2,1).reshape(B,C,16,16),up_m)
        # clip_image_embeds = torch.cat((clip_image_embeds[:,0:1,:],clip_image_embeds_up.reshape(B,C,-1).permute(0,2,1)),dim=1)

        image_prompt_embeds = self.image_proj_model2(clip_image_embeds).to(dtype=torch.float32)
        uncond_clip_image_embeds = self.image_encoder(
            torch.zeros_like(clip_image), output_hidden_states=True
        ).hidden_states[-2]
        # uncond_clip_image_embeds_up = self.upsample_flow(uncond_clip_image_embeds[:,1:,:].permute(0,2,1).reshape(B,C,16,16),up_m)
        # uncond_clip_image_embeds = torch.cat((uncond_clip_image_embeds[:,0:1,:],uncond_clip_image_embeds_up.reshape(B,C,-1).permute(0,2,1)),dim=1)
        uncond_image_prompt_embeds = self.image_proj_model2(uncond_clip_image_embeds)
        if return_hidden:
            return image_prompt_embeds, uncond_image_prompt_embeds, clip_image_embeds
        return image_prompt_embeds, uncond_image_prompt_embeds
    
    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
                
    def forward_corr(self, pil_image=None, up_ft_index=None,uncond_style_prompt_flag=None, style_prompt_flag=None,clip_image_embeds=None,prompt=None,negative_prompt=None,scale=1.0,num_samples=1,seed=None,guidance_scale=7.5,num_inference_steps=30,**kwargs,):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "a photo of a portrait"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds, clip_image_embeds_ref = self.get_image_embeds_corr(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds,  return_hidden=True
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1).to("cuda")
        # image_prompt_embeds = torch.zeros((1,257,768)).to("cuda")
        # uncond_image_prompt_embeds = torch.zeros((1,257,768)).to("cuda")
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)

        style_prompt_flag = style_prompt_flag.flatten(2,3).permute(0,2,1)
        uncond_style_prompt_flag = uncond_style_prompt_flag.flatten(2,3).permute(0,2,1)
        prompt_embeds = torch.cat([style_prompt_flag,prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([uncond_style_prompt_flag,negative_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)

        unet_ft_all = self.pipe.extract_corr(
            up_ft_indices=[up_ft_index],
            ref_image = pil_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            train_loss=True,
            **kwargs,
        )
        unet_ft = unet_ft_all['up_ft'][up_ft_index] # ensem, c, h, w
        unet_ft = unet_ft.mean(0, keepdim=True) # 1,c,h,w
        return unet_ft
    
    def forward(self, ref_image=None, pil_image=None, uncond_style_prompt_flag=None, style_prompt_flag=None,clip_image_embeds=None,prompt=None,negative_prompt=None,scale=1.0,num_samples=1,seed=None,guidance_scale=7.5,num_inference_steps=30,**kwargs,):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "a photo of a portrait"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts
        if style_prompt_flag[0,0,0,0] != 0:
            mode = "style"
        else:
            mode = "corr"

        if mode=="style":
            image_prompt_embeds, uncond_image_prompt_embeds, clip_image_embeds_ref = self.get_image_embeds_style(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds, return_hidden=True
            )
        else:
            image_prompt_embeds, uncond_image_prompt_embeds, clip_image_embeds_ref = self.get_image_embeds_corr(
                pil_image=pil_image, clip_image_embeds=clip_image_embeds, return_hidden=True
            )          
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1).to("cuda")

        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
        prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        generator = None

        style_prompt_flag = style_prompt_flag.flatten(2,3).permute(0,2,1)
        uncond_style_prompt_flag = uncond_style_prompt_flag.flatten(2,3).permute(0,2,1)
        prompt_embeds = torch.cat([style_prompt_flag,prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([uncond_style_prompt_flag,negative_prompt_embeds], dim=1)

        loss_st = self.pipe(
            ref_image = ref_image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            train_loss=True,
            **kwargs,
        )
        return loss_st
    
    @torch.no_grad()
    def generate(
        self,
        pil_image=None,
        ref_image=None,
        image_sd=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        uncond_style_prompt_flag=None,
        style_prompt_flag=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)
        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)
        if prompt is None:
            prompt = "a photo of an artistic portrait"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_style(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        image_prompt_embeds_input, uncond_image_prompt_embeds = self.get_image_embeds_style(
            pil_image=ref_image, clip_image_embeds=clip_image_embeds
        )
        sty_scale = 1.0
        image_prompt_embeds = sty_scale*image_prompt_embeds + (1-sty_scale) * image_prompt_embeds_input
        concat = False
        if concat:
            image_prompt_embeds = torch.cat((image_prompt_embeds,image_prompt_embeds_input),dim=1)
            uncond_image_prompt_embeds = torch.cat((uncond_image_prompt_embeds,uncond_image_prompt_embeds),dim=1)

        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1).to("cuda")
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1).to("cuda")
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        style_prompt_flag = style_prompt_flag.flatten(2,3).permute(0,2,1).to("cuda")
        uncond_style_prompt_flag = uncond_style_prompt_flag.flatten(2,3).permute(0,2,1).to("cuda")
        prompt_embeds = torch.cat([style_prompt_flag,prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([uncond_style_prompt_flag,negative_prompt_embeds], dim=1)
        generator = get_generator(seed, self.device)
        images = self.pipe(
            ref_image = ref_image,
            prompt_embeds=prompt_embeds,
            image_sd=image_sd,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        return images
    
    @torch.no_grad()
    def inverse(
        self,
        pil_image=None,
        ref_image=None,
        image_sd=None,
        clip_image_embeds=None,
        prompt=None,
        negative_prompt=None,
        uncond_style_prompt_flag=None,
        style_prompt_flag=None,
        scale=1.0,
        num_samples=1,
        seed=None,
        guidance_scale=7.5,
        num_inference_steps=30,
        **kwargs,
    ):
        self.set_scale(scale)

        if pil_image is not None:
            num_prompts = 1 if isinstance(pil_image, Image.Image) else len(pil_image)
        else:
            num_prompts = clip_image_embeds.size(0)

        if prompt is None:
            prompt = "a photo of an artistic portrait"
        if negative_prompt is None:
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"

        if not isinstance(prompt, List):
            prompt = [prompt] * num_prompts
        if not isinstance(negative_prompt, List):
            negative_prompt = [negative_prompt] * num_prompts

        image_prompt_embeds, uncond_image_prompt_embeds = self.get_image_embeds_style(
            pil_image=pil_image, clip_image_embeds=clip_image_embeds
        )
        bs_embed, seq_len, _ = image_prompt_embeds.shape
        image_prompt_embeds = image_prompt_embeds.repeat(1, num_samples, 1)
        image_prompt_embeds = image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1).to("cuda")
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.repeat(1, num_samples, 1)
        uncond_image_prompt_embeds = uncond_image_prompt_embeds.view(bs_embed * num_samples, seq_len, -1).to("cuda")
        with torch.inference_mode():
            prompt_embeds_, negative_prompt_embeds_ = self.pipe.encode_prompt(
                prompt,
                device=self.device,
                num_images_per_prompt=num_samples,
                do_classifier_free_guidance=True,
                negative_prompt=negative_prompt,
            )
            prompt_embeds = torch.cat([prompt_embeds_, image_prompt_embeds], dim=1)
            negative_prompt_embeds = torch.cat([negative_prompt_embeds_, uncond_image_prompt_embeds], dim=1)
        style_prompt_flag = style_prompt_flag.flatten(2,3).permute(0,2,1).to("cuda")
        uncond_style_prompt_flag = uncond_style_prompt_flag.flatten(2,3).permute(0,2,1).to("cuda")
        prompt_embeds = torch.cat([style_prompt_flag,prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([uncond_style_prompt_flag,negative_prompt_embeds], dim=1)

        generator = get_generator(seed, self.device)
        images = self.pipe.call_inversrion(
            ref_image = ref_image,
            prompt_embeds=prompt_embeds,
            image_sd=image_sd,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            **kwargs,
        ).images
        return images
