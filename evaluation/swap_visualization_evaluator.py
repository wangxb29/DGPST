import os
from PIL import Image
import numpy as np
import torch
from evaluation import BaseEvaluator
import util


class SwapVisualizationEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument("--swap_num_columns", type=int, default=4,
                            help="number of images to be shown in the swap visualization grid. Setting this value will result in 4x4 swapping grid, with additional row and col for showing original images.")
        parser.add_argument("--swap_num_images", type=int, default=16,
                            help="total number of images to perform swapping. In the end, (swap_num_images / swap_num_columns) grid will be saved to disk")
        return parser

    def gather_images(self, dataset):
        all_images = []
        all_masks = []
        num_images_to_gather = max(self.opt.swap_num_columns, self.opt.num_gpus)
        exhausted = False
        while len(all_images) < num_images_to_gather:
            try:
                data = next(dataset)
            except StopIteration:
                print("Exhausted the dataset at %s" % (self.opt.dataroot))
                exhausted = True
                break
            for i in range(data["real_A"].size(0)):
                all_images.append(data["real_A"][i:i+1])
                all_masks.append(data["mask"][i:i+1])
                if len(all_images) >= num_images_to_gather:
                    break
        if len(all_images) == 0:
            return None, None, True
        return all_images, exhausted, all_masks

    def generate_mix_grid(self, model, images, masks):

        def put_img(img, canvas, row, col):
            h, w = img.shape[0], img.shape[1]
            start_x = int(self.opt.load_size * col + (self.opt.load_size - w) * 0.5)
            start_y = int(self.opt.load_size * row + (self.opt.load_size - h) * 0.5)  
            canvas[start_y:start_y + h, start_x: start_x + w] = img
        grid_w = self.opt.load_size * (4 + 1)
        grid_h = self.opt.load_size * (4 + 1)
        grid_img = np.ones((grid_h, grid_w, 3), dtype=np.uint8)
        grid_img_warp = np.ones((grid_h, grid_w, 3), dtype=np.uint8)
        #images_np = util.tensor2im(images, tile=False)
        for i, image in enumerate(images):

            image_np = util.tensor2im(image, normalize=True, tile=False)[0]
            image_nomask = util.tensor2im(image, normalize=True, tile=False)[0]
            
            #image_np = util.tensor2im(image, normalize=False, tile=False)[0]
            put_img(image_nomask, grid_img, 0, i + 1)
            put_img(image_np, grid_img, i + 1, 0)
            put_img(image_nomask, grid_img_warp, 0, i + 1)
            put_img(image_np, grid_img_warp, i + 1, 0)

        for i in range(4):
            mix_row1, warp_row1 = model(images[0], images[i], masks[i], masks[0], command='generate')  
            mix_row2, warp_row2 = model(images[1], images[i], masks[i], masks[1], command='generate')  
            mix_row3, warp_row3 = model(images[2], images[i], masks[i], masks[2], command='generate')  
            mix_row4, warp_row4 = model(images[3], images[i], masks[i], masks[3], command='generate')  
            mix_row = torch.cat((mix_row1,mix_row2,mix_row3,mix_row4),dim=0)
            warp_row = torch.cat((warp_row1,warp_row2,warp_row3,warp_row4),dim=0)
            #mix_row = model(mix_row, command='Cluster')
            mix_row = util.tensor2im(mix_row, normalize=True, tile=False)
            warp_row = util.tensor2im(warp_row, normalize=True, tile=False)
            for j, mix in enumerate(mix_row):
                put_img(mix, grid_img, i + 1, j + 1)
            # mix_row1 = util.tensor2im(mix_row1, normalize=True, tile=False)
            for j, mix in enumerate(warp_row):
                put_img(mix, grid_img_warp, i + 1, j + 1)
        # data = corrmatrix[:,0:100,0:100].reshape(-1,100).cpu().numpy()
        # np.savetxt("/home/xinbo/corr.txt",data ,fmt="%s",delimiter=",")

        final_grid = Image.fromarray(grid_img)
        final_grid_warp = Image.fromarray(grid_img_warp)
        return final_grid, final_grid_warp

    def evaluate(self, model, dataset, nsteps):
        nsteps = self.opt.resume_iter if nsteps is None else str(round(nsteps / 1000)) + "k"
        savedir = os.path.join(self.output_dir(), "%s_%s" % (self.target_phase, nsteps))
        os.makedirs(savedir, exist_ok=True)
        webpage_title = "Swap Visualization of %s. iter=%s. phase=%s" % \
                        (self.opt.name, str(nsteps), self.target_phase)
        webpage = util.HTML(savedir, webpage_title)
        num_repeats = int(np.ceil(self.opt.swap_num_images / max(self.opt.swap_num_columns, self.opt.num_gpus)))
        num_repeats = 1
        for i in range(num_repeats):
            images, should_break, masks = self.gather_images(dataset)
            if images is None:
                break
            mix_grid, warp_grid = self.generate_mix_grid(model, images, masks)
            webpage.add_images([mix_grid], ["%04d.png" % i])
            webpage.add_images([warp_grid], ["%04d_warp.png" % i])
            if should_break:
                break
        webpage.save()
        return {}
