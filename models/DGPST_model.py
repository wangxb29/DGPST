import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import exp, floor, log2, sqrt, pi
from models import BaseModel
from models.networks.function import calc_mean_std, nor_mean_std, nor_mean, calc_cov
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
from diffusers import ControlNetModel
from ip_adapter.ip_adapter_control import IPAdapter2
from models.networks.dift_sd import MyUNet2DConditionModel
from models.pipeline_dgpst import DGPSTPipeline
from src.schedulers.ddim_scheduler import MyDDIMScheduler
from src.eunms import Model_Type, Scheduler_Type
from src.config import RunConfig
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation, SegformerFeatureExtractor
from models.networks.drawingmodel import Generator
import util

def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    #   [batch,channel,H,W]
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
 
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
    
def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    LH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HL = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)
    HH = net(in_channels, in_channels,
             kernel_size=2, stride=2, padding=0, bias=False,
             groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1)
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1)

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels)

    def forward(self, x):
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)
    
class WaveUnpool(nn.Module):
    def __init__(self, in_channels, option_unpool='sum'):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.option_unpool = option_unpool
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH=None, HL=None, HH=None, original=None):
        if LH==None:
            return self.LL(LL)
        if self.option_unpool == 'sum':
            return self.LL(LL) + self.LH(LH) + self.HL(HL) + self.HH(HH)
        elif self.option_unpool == 'cat5' and original is not None:
            return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)
        else:
            raise NotImplementedError
        
class DGPSTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        BaseModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--training_stage", default=2, type=int)
        parser.add_argument("--lambda_Maskwarp", default=10.0, type=float)
        parser.add_argument("--lambda_Cycwarp", default=1.0, type=float)
        parser.add_argument('--t', default=0, type=int, 
                            help='time step for diffusion, choose from range [0, 1000]')
        parser.add_argument('--up_ft_index', default=2, type=int, choices=[0, 1, 2 ,3],
                            help='which upsampling block of U-Net to extract the feature map')
        parser.add_argument('--prompt_content', default='a photo of a portrait', type=str)
        parser.add_argument('--prompt_style', default='a photo of a portrait', type=str)
        parser.add_argument('--prompt_output', default='a photo of a portrait', type=str)
        parser.add_argument('--auto_mask', default=False, type=bool)
        parser.add_argument('--region_style', default=False, type=bool)
        parser.add_argument('--gamma_interpolate', default=1, type=int)
        parser.add_argument('--post_process', default=False, type=bool) 

        return parser

    def initialize(self):
        torch.distributed.init_process_group(
            backend='gloo',
            init_method='env://',
            rank=self.opt.local_rank,
            world_size=self.opt.num_gpus,
            )
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()
        device = torch.device("cuda", local_rank)

        image_encoder_path = "ip_adapter/models/image_encoder"
        ip_ckpt = "ip_adapter/models/ip-adapter-full-face_sd15.safetensors"
        unet = MyUNet2DConditionModel.from_pretrained(
        "runwayml/stable-diffusion-v1-5", subfolder="unet"
        )
        if os.path.exists("checkpoints/CelebA_diff69/diffusion_pytorch_model.safetensors"):
            controlnet = ControlNetModel.from_pretrained("D:\projects\Diffswapping-4\checkpoints\CelebA_diff69")
        else:
            controlnet = ControlNetModel.from_unet(unet)
        pipe = DGPSTPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, unet=unet
        )
        pipe.scheduler = MyDDIMScheduler.from_config(pipe.scheduler.config)
        config = RunConfig(model_type = Model_Type.SD15,
                        num_inference_steps = 50,
                        num_inversion_steps = 50,
                        num_renoise_steps = 1,
                        scheduler_type = Scheduler_Type.DDIM,
                        perform_noise_correction = False,
                        seed = 7865
                        )
        pipe.cfg = config
        self.ip_model = IPAdapter2(pipe, image_encoder_path, ip_ckpt, device)

        self.l1_loss = torch.nn.L1Loss()
        self.loss_fn_alex = lpips.LPIPS(net='alex')
        if (not self.opt.isTrain) or self.opt.continue_train:
            self.load()
        # self.ip_model.load_ip_adapter()
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.wav = WavePool(3)
        self.wavunpool = WaveUnpool(3)
        self.wav4 = WavePool(4)
        self.wavunpool4 = WaveUnpool(4)

        # self.image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
        # self.model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
        self.image_processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        self.model.to(device)
        
        exclude_keys = ["ip_model.image_encoder", "loss_fn_alex"]  
        # 创建过滤后的 state_dict
        self.filtered_state_dict = {
            k: v for k, v in self.state_dict().items() 
            if not any(ex in k for ex in exclude_keys)
        }

    def per_gpu_initialize(self):
        pass

    def compute_generator_losses(self, real, mask): 
        real_A = real[0:1]
        real_B = real[1:2]
        losses = {}
        metrics = {}

        _, LH, HL, HH = self.wav(real_B)
        control_B = torch.cat((LH,HL,HH),dim=1)
        control_B = F.interpolate(control_B,scale_factor=2,mode='nearest')     
        _, LH, HL, HH = self.wav(real_A)
        control_A = torch.cat((LH,HL,HH),dim=1)
        control_A = F.interpolate(control_A,scale_factor=2,mode='nearest')

        with torch.no_grad():
            real_B_img = (real_B/2 + 0.5) * 255
            real_A_img = (real/2 + 0.5) * 255
            real_B_img = Image.fromarray(real_B_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
            real_A_img = Image.fromarray(real_A_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
        
        if self.opt.training_stage==1:
            style_prompt_flag = torch.zeros((1,768,1,1),device=real.device).detach()
            uncond_style_prompt_flag = torch.zeros((1,768,1,1),device=real.device).detach()
            fea_inputB, _ = self.ip_model.forward_corr(pil_image=real_B_img, up_ft_index=self.opt.up_ft_index, uncond_style_prompt_flag=uncond_style_prompt_flag, style_prompt_flag=style_prompt_flag, image=control_B, image_sd = real_B_img,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=30, mask=None, seed=None)
            fea_inputA, _ = self.ip_model.forward_corr(pil_image=real_A_img, up_ft_index=self.opt.up_ft_index, uncond_style_prompt_flag=uncond_style_prompt_flag, style_prompt_flag=style_prompt_flag, image=control_A, image_sd = real_A_img,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=30, mask=None, seed=None)
            Q1 = fea_inputA.flatten(-2, -1).permute(0,2,1)
            K1 = fea_inputB.flatten(-2, -1)
            Q1 = Q1 - Q1.mean(dim=2,keepdim=True)
            Q1_norm = (torch.norm(Q1, 2, 2, keepdim=True)+sys.float_info.epsilon)
            Q1 = Q1 / Q1_norm
            K1 = K1 - K1.mean(dim=1,keepdim=True)
            K1_norm = (torch.norm(K1, 2, 1, keepdim=True)+sys.float_info.epsilon)
            K1 = K1 / K1_norm
            A1 = torch.bmm(Q1, K1)/0.01
            A1_T = torch.softmax(A1.permute(0,2,1),dim=-1)
            A1 = torch.softmax(A1,dim=-1)

            if self.opt.lambda_Maskwarp > 0.0:
                mask_warp = self.warp(mask[1:2],A1,mask[0:1])
                losses["Mask_warp"] = self.l1_loss(mask_warp, mask[0:1]) * self.opt.lambda_Maskwarp

            if self.opt.lambda_Cycwarp > 0.0:
                real_A_w = self.warp(real,A1_T,real) 
                real_A_ww = self.warp(real_A_w,A1,real)
                losses["loss_cycwarp"] = self.loss_fn_alex(real_A_ww,real) * self.opt.lambda_Cycwarp
        else:
            style_prompt_flag = torch.ones((1,768,1,1),device=real.device).detach()
            uncond_style_prompt_flag = torch.ones((1,768,1,1),device=real.device).detach()

        loss0 = self.ip_model.forward(ref_image=real_A_img, pil_image=real_A_img, uncond_style_prompt_flag=uncond_style_prompt_flag, style_prompt_flag=style_prompt_flag, image=control_A, image_sd = real_A_img, controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=50, mask=None, seed=None)
        losses["loss_diff"] = loss0 

        return losses, metrics

    def _mask_labels(self, mask_np):
        label_size = 19
        labels = torch.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        
        return labels.unsqueeze(0)
    
    def generate(self, real_B, real_A, mask, mask_ref, seed=None, extract_features=False):    

        LL, LH, HL, HH = self.wav(real_B)
        control_B = torch.cat((LH,HL,HH),dim=1)
        control_B = F.interpolate(control_B,scale_factor=2,mode='bilinear')
        LL, LH, HL, HH = self.wav(real_A)
        control_A = torch.cat((LH,HL,HH),dim=1)
        control_A = F.interpolate(control_A,scale_factor=2,mode='bilinear')
        real_B_img = (real_B/2 + 0.5) * 255
        real_A_img = (real_A/2 + 0.5) * 255
        real_B_img = Image.fromarray(real_B_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8))
        real_A_img = Image.fromarray(real_A_img[0].permute(1,2,0).cpu().numpy().astype(np.uint8))

        style_prompt_flag = torch.zeros((1,768,1,1),device=real_A.device).detach()
        uncond_style_prompt_flag = torch.zeros((1,768,1,1),device=real_A.device).detach()

        fea_inputB = self.ip_model.forward_corr(prompt=self.opt.prompt_style,pil_image=real_B_img, up_ft_index=self.opt.up_ft_index, uncond_style_prompt_flag=uncond_style_prompt_flag, style_prompt_flag=style_prompt_flag, image=control_B, image_sd = real_B_img,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=30, mask=None, seed=None)
        fea_inputA = self.ip_model.forward_corr(prompt=self.opt.prompt_content,pil_image=real_A_img, up_ft_index=self.opt.up_ft_index, uncond_style_prompt_flag=uncond_style_prompt_flag, style_prompt_flag=style_prompt_flag, image=control_A, image_sd = real_A_img,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=30, mask=None, seed=None)
        B,C,H,W = real_A.size()
        B,C,H2,W2 = real_B.size()
        Q1 = fea_inputA.flatten(-2, -1).permute(0,2,1)
        K1 = fea_inputB.flatten(-2, -1)
        Q1 = Q1 - Q1.mean(dim=2,keepdim=True)
        Q1_norm = (torch.norm(Q1, 2, 2, keepdim=True)+sys.float_info.epsilon)
        Q1 = Q1 / Q1_norm
        K1 = K1 - K1.mean(dim=1,keepdim=True)
        K1_norm = (torch.norm(K1, 2, 1, keepdim=True)+sys.float_info.epsilon)
        K1 = K1 / K1_norm
        A1 = torch.bmm(Q1, K1)

        if mask==None and self.opt.auto_mask:
            inputs = self.image_processor(images=real_A_img, return_tensors="pt").to(real_A.device)
            outputs = self.model(**inputs)
            mask = outputs.logits 
            mask = self._mask_labels(mask.argmax(dim=1)[0]).to(real_A.device)
            inputs = self.image_processor(images=real_B_img, return_tensors="pt").to(real_A.device)
            outputs = self.model(**inputs)
            mask_ref = outputs.logits 
            mask_ref = self._mask_labels(mask_ref.argmax(dim=1)[0]).to(real_A.device)
            mask = F.interpolate(mask,size=(H,W),mode='bilinear')
            mask_ref = F.interpolate(mask_ref,size=(H2,W2),mode='bilinear')
        if mask!=None and mask_ref!=None:
            mask_d = F.interpolate(mask,scale_factor=0.125,mode='bilinear')
            mask_ref = F.interpolate(mask_ref,scale_factor=0.125,mode='bilinear')
            mask_ce = torch.argmax(mask_d, dim=1).reshape(B,-1,1)
            mask_ref_ce = torch.argmax(mask_ref, dim=1).reshape(B,1,-1)
            accury =  mask_ce - mask_ref_ce
            A1[abs(accury)>=0.25] = -1       

        A1 = A1 / 0.01
        A1 = torch.softmax(A1,dim=-1)
        real_B_w = self.warp(real_B,A1,real_A)
        real_B_w_tensor = real_B_w.clone()
        return real_B_w_tensor, real_B_w_tensor
    
        if self.opt.region_style:
            new_mask = mask[:,13:14] #hair
            #new_mask = mask[:,11:12] + mask[:,12:13] #lips
            #new_mask = mask[:,1:2] + mask[:,2:3] +mask[:,6:7]+mask[:,7:8]+mask[:,8:9]+mask[:,9:10]+mask[:,17:18] #face
            real_B_w = real_B_w * new_mask + real_A * (1-new_mask)

        real_B_w = (real_B_w/2 + 0.5) * 255
        real_B_w = Image.fromarray(real_B_w[0].permute(1,2,0).cpu().numpy().astype(np.uint8))

        # AdaIN-wavelet
        style_prompt_flag = - torch.ones((1,768,1,1),device=real_A.device).detach()
        uncond_style_prompt_flag = - torch.ones((1,768,1,1),device=real_A.device).detach()
        inv_latent = self.ip_model.inverse(latents=None, prompt=self.opt.prompt_content,pil_image=real_A_img, ref_image=real_A_img, uncond_style_prompt_flag=uncond_style_prompt_flag,style_prompt_flag=style_prompt_flag, image=control_A, image_sd = real_A,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=10, mask=None, seed=seed)
        inv_latent_ref = self.ip_model.inverse(latents=None, prompt=self.opt.prompt_style,pil_image=real_B_w, ref_image=real_B_w, uncond_style_prompt_flag=uncond_style_prompt_flag,style_prompt_flag=style_prompt_flag, image=control_A, image_sd = real_B_w,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=10, mask=None, seed=seed) 
        inv_latent_ada = adaptive_instance_normalization(inv_latent,inv_latent_ref)
        LL, _, _, _ = self.wav4(inv_latent_ref)
        _, LH, HL, HH = self.wav4(inv_latent_ada)
        inv_latent_com = self.wavunpool4(LL, LH, HL, HH)       

        # interpolate
        inv_latent_com = self.opt.gamma_interpolate * inv_latent_com + (1-self.opt.gamma_interpolate) * inv_latent

        style_prompt_flag = torch.ones((1,768,1,1),device=real_A.device).detach()
        uncond_style_prompt_flag = torch.ones((1,768,1,1),device=real_A.device).detach()
        pred = self.ip_model.generate(latents=inv_latent_com,prompt=self.opt.prompt_output, pil_image=real_B_w, ref_image=real_B_w, uncond_style_prompt_flag=uncond_style_prompt_flag,style_prompt_flag=style_prompt_flag, image=control_A, image_sd = real_A,  controlnet_conditioning_scale=0.9, num_samples=1, num_inference_steps=30, mask=None, seed=seed)

        pred= (pred-0.5)*2
        if self.opt.post_process:            
            from photo_gif import GIFSmoothing
            target = real_A / 2 + 0.5
            out = pred / 2 + 0.5
            p_pro = GIFSmoothing(r=8, eps=(0.02 * 255) ** 2)
            new_out = torch.zeros_like(out,device=out.device)                
            out = util.tensor2im(out,tile=False)              
            target = util.tensor2im(target,tile=False)
            for i,(ori, wai) in enumerate(zip(target, out)):
                wai = Image.fromarray(wai)
                ori = Image.fromarray(ori)                          
                wai = p_pro.process(wai, ori)      
                wai = self.to_tensor(wai)
                ori = self.to_tensor(ori)
                wai = (wai-0.5)*2
                ori = (ori-0.5)*2
                new_out[i] = wai
            return new_out, real_B_w_tensor
        return pred, real_B_w_tensor
    
    def get_parameters_for_mode(self, mode):
        if mode == "Diff":
            return list(self.ip_model.parameters()) + list(self.ip_model.pipe.controlnet.parameters()) 
    
    def warp(self, fea, corr, fea_in=None):
        b,c,h,w = fea.size()
        _, _, hi, wi = fea_in.size() 
        l = h * w
        B,H,W = corr.size()
        if W != l:                                                                                            
            s = int((l/W) ** 0.5)
            si = int((hi*wi/H) ** 0.5)
            unfold = False
            if unfold:
                feas = F.unfold(fea, s, stride=s)
            else:
                feas = F.interpolate(fea,size=(h//s,w//s),mode='bilinear')
                feas = feas.view(b,c,-1)
            feas = feas.permute(0,2,1).contiguous()
            warp_fea = torch.matmul(corr.to(dtype=torch.float32), feas.to(dtype=torch.float32))
            warp_fea = warp_fea.permute(0,2,1).contiguous()
            if unfold==False:
                warp_fea = warp_fea.view(b,c,int(hi/si),int(wi/si))
                warp_fea = F.interpolate(warp_fea,scale_factor=si,mode='bilinear')
            else:
                warp_fea = F.fold(warp_fea, (h,w) ,s, stride=s)
            # warp_fea = F.fold(warp_fea, (h,warp_fea.size(2)*warp_fea.size(3)//h) ,s, stride=s)
            return warp_fea
        fea = fea.view(b,c,-1).permute(0,2,1).contiguous()
        warp_feat_f = torch.matmul(corr, fea)
        warp_feat = warp_feat_f.permute(0, 2, 1).view(b,c,h,w).contiguous()
        return warp_feat 
    
    def save(self, total_steps_so_far):
        savedir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        checkpoint_name = "%dk_checkpoint.pth" % (total_steps_so_far // 1000)
        savepath = os.path.join(savedir, checkpoint_name)
        torch.save(self.filtered_state_dict, savepath)
        sympath = os.path.join(savedir, "latest_checkpoint.pth")
        sympath = os.path.normpath(sympath)
        if os.path.exists(sympath):
            os.remove(sympath)        
            print("remove!")
        savepath = os.path.normpath(savepath)
        os.symlink(savepath, sympath)

        controlnet_savepath = os.path.join(savedir, "%dk_controlnet" % (total_steps_so_far // 1000))
        self.ip_model.pipe.controlnet.save_pretrained(controlnet_savepath)
        self.ip_model.pipe.controlnet.save_pretrained(savedir)
        # sympath = os.path.join(controlnet_savepath, "latest_checkpoint.pth")
        # if os.path.exists(sympath):
        #     os.remove(sympath)
        # os.symlink(checkpoint_name, sympath)   