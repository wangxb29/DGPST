import os
import cv2
import numpy as np
from utils import make_folder
import numpy as np
np.set_printoptions(threshold=np.inf)
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

label_map = {
    'skin': 1,
    'nose': 2,
    'eye_g': 3,
    'l_eye': 4,
    'r_eye': 5,
    'l_brow': 6,
    'r_brow': 7,
    'l_ear': 8,
    'r_ear': 9,
    'mouth': 10,
    'u_lip': 11,
    'l_lip': 12,
    'hair': 13,
    'hat': 14,
    'ear_r': 15,
    'neck_l': 16,
    'neck': 17,
    'cloth': 18
}

folder_base = '../CelebAMask-HQ/CelebAMask-HQ-mask-anno'
folder_save = '../CelebAMask-HQ/CelebAMask-HQ-img-mask-19'
img_num = 30000 


make_folder(folder_save)

for k in range(img_num):
    folder_num = k // 2000 
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            #print (label, idx+1)
            im = cv2.imread(filename)
            im = im[:, :, 0]  
            im_base[im != 0] = label_map[label]
    filename_save = os.path.join(folder_save, str(k) + '.png')
    print (filename_save)
    cv2.imwrite(filename_save, im_base) # 保存图片