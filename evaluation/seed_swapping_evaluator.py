import os
import cv2
import numpy as np
import torch

import torchvision.transforms as transforms
from PIL import Image
from evaluation import BaseEvaluator
from data.base_dataset import get_transform
import util


class seedSwappingEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--input_structure_image", required=True, type=str)
        parser.add_argument("--input_texture_image", required=True, type=str)
        parser.add_argument("--input_structure_mask", required=False, type=str)
        parser.add_argument("--input_texture_mask", required=False, type=str)
        parser.add_argument("--texture_mix_alphas", type=float, nargs='+',
                            default=[1.0],
                            help="Performs interpolation of the texture image."
                            "If set to 1.0, it performs full swapping."
                            "If set to 0.0, it performs direct reconstruction"
                            )
        
        opt, _ = parser.parse_known_args()
        dataroot = os.path.dirname(opt.input_structure_image)
        
        # dataroot and dataset_mode are ignored in SimpleSwapplingEvaluator.
        # Just set it to the directory that contains the input structure image.
        parser.set_defaults(dataroot=dataroot, dataset_mode="imagefolder")
        
        return parser
    
    def load_image(self, path):
        path = os.path.expanduser(path)
        img = Image.open(path).convert('RGB')
        transform = get_transform(self.opt)
        tensor = transform(img).unsqueeze(0)
        return tensor
    
    def _mask_labels(self, mask_np):
        label_size = 19
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        return labels
    
    def load_mask(self, path):
        path = os.path.expanduser(path)
        mask = Image.open(path).convert('L')
        mask_np = np.array(mask)
        labels = self._mask_labels(mask_np)
        mask_tensor = torch.tensor(labels, dtype=torch.float).unsqueeze(0)
        return mask_tensor
    
    def evaluate(self, model, dataset, nsteps=None):
        iterations = 4096
        dummy_input = torch.rand(1, 3, 512, 512).to('cuda')
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        # 初始化一个时间容器
        print(self.opt.input_structure_image)
        print(self.opt.input_texture_image)
        structure_image = self.load_image(self.opt.input_structure_image)
        texture_image = self.load_image(self.opt.input_texture_image)    
        structure_mask = None
        texture_mask = None
        if self.opt.input_structure_mask!=None:    
            structure_mask = self.load_mask(self.opt.input_structure_mask).cuda()
            texture_mask = self.load_mask(self.opt.input_texture_mask).cuda()
        print('warm up ...\n')
        # with torch.no_grad():
        #     for _ in range(50):
        #         _ = model(dummy_input, command="encode")
        times = np.zeros((iterations, 1))    
        with torch.no_grad():
            for iter in range(iterations):
                starter.record() 
                os.makedirs(self.output_dir(), exist_ok=True)
                output_image, _ = model(texture_image.cuda(), structure_image.cuda(), structure_mask, texture_mask, seed=iter, command='generate')  
                ender.record()
                # 同步GPU时间
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) # 计算时间
                times[iter] = curr_time
                
                output_name = "%s_%s_%d.png" % (
                    os.path.splitext(os.path.basename(self.opt.input_structure_image))[0],
                    os.path.splitext(os.path.basename(self.opt.input_texture_image))[0],
                    iter,
                )
                output_path = os.path.join(self.output_dir(), output_name)
                output_image = transforms.ToPILImage()((output_image[0].clamp(-1.0, 1.0) + 1.0) * 0.5)
                output_image.save(output_path)
                print("Saved at " + output_path)     

        mean_time = times.mean().item()
        print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000/mean_time))
        return {}
