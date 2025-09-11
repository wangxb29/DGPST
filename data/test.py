import random
from PIL import Image
import numpy as np
import torch
from skimage.color import rgb2lab, lab2rgb

def to_mytensor(pic):
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


def normalize(tensor, mean, std):
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

A_img = Image.open("/home/xinbo/projects/contrastive_swapping_autoencoder/testphotos/ffhq512/ffhq1/structure/00230.png").convert('RGB')
labA = rgb2lab(A_img)
labA = to_mytensor(labA)
#labA = to_mytensor(A_img)
# labA[0:3, :, :] = normalize(labA[0:3, :, :], (0, 0, 0), (255, 255, 255))
# labA[0:3, :, :] = normalize(labA[0:3, :, :], (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
labA[0:3, :, :] = torch.round(normalize(labA[0:3, :, :], (50, 0, 0), (50, 100, 100)),decimals=4)
print(labA)