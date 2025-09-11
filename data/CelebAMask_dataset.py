import random

import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torch
from util.MattingLaplacian import compute_laplacian

#from skimage.color import rgb2lab, lab2rgb

class CelebAMaskDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = opt.dataroot
        self.dir_B = opt.dataroot2
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)
        self.transform = get_transform(self.opt, grayscale=False, convert=False)
        self.transform2 = get_transform(self.opt, convert=False)
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.color_map = {
            0: [  0,   0,   0],
            1: [ 0,0,205],
            2: [132,112,255],
        }
        #self.use_lap = False
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        return self.getitem_by_path(A_path, B_path)

    def getitem_by_path(self, A_path, B_path):
        try:
            A_img = Image.open(A_path).convert('RGB')
            B_img = Image.open(B_path).convert('L')
        except OSError as err:
            print(err)
            return self.__getitem__(random.randint(0, len(self) - 1))

        A = self.transform(A_img)
        B = self.transform2(B_img)   
        A = self.to_tensor(A)
        A = (A-0.5) * 2
        mask_np = np.array(B)
        labels = self._mask_labels(mask_np)
        mask_tensor = torch.tensor(labels, dtype=torch.float)

        return {'real_A': A, 'mask': mask_tensor, 'path_A': A_path}

    def __len__(self):
        return self.A_size
    
    def _mask_labels(self, mask_np):
        label_size = 19
        labels = np.zeros((label_size, mask_np.shape[0], mask_np.shape[1]))
        for i in range(label_size):
            labels[i][mask_np==i] = 1.0
        
        return labels

    def to_mytensor(self, pic):
        """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

        See ``ToTensor`` for more details.

        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pic_arr = np.array(pic)
        if pic_arr.ndim == 2:
            pic_arr = pic_arr[..., np.newaxis]
        img = torch.from_numpy(pic_arr.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            return img.float()  # no normalize .div(255)
        else:
            return img

    def normalize(self, tensor, mean, std):
        """Normalize a tensor image with mean and standard deviation.

        See ``Normalize`` for more details.

        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            mean (sequence): Sequence of means for each channel.
            std (sequence): Sequence of standard deviations for each channely.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # if not _is_tensor_image(tensor):
        #     raise TypeError("tensor is not a torch image.")
        # TODO: make efficient
        if tensor.size(0) == 1:
            tensor.sub_(mean).div_(std)
        else:
            for t, m, s in zip(tensor, mean, std):
                t.sub_(m).div_(s)
        return tensor