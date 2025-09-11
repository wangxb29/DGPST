# DG-PST: Domain Generalizable Portrait Style Transfer
Official implementation of "Domain Generalizable Portrait Style Transfer" (Acceped to ICCV 2025)

## Installation
Set up the python environment
``` python
conda create -n dgpst python=3.8
conda activate dgpst

pip install -r requirements.txt
```

## Set up dataset
Please download [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset.

## Training
Please set the ```dataroot``` path as your image path and ```dataroot2``` path to your mask path in ```./experiments/CelebA_launcher.py```, and run
``` python
python -m experiments CelebA train CelebAMaskHQ_default
```

## Testing
To perform style transfer between two given images, you can put the pretrained model in ```./checkpoints/CelebA_default/`` and run 
``` python
CUDA_VISIBLE_DEVICES=0 --nproc_per_node=1 --master_port='29501' test.py \ 
--name CelebA_default \ 
--evaluation_metrics seed_swapping \ 
--preprocess scale_shortside \
--load_size 512 \
--model DGPST \
--input_structure_image /path/to/your/content/image \
--input_texture_image /path/to/your/style/image
```
To perform style transfer between image folders, please set the ```dataroot``` and ```checkpoints_dir``` path in ```./experiments/CelebA_launcher.py```, and put the content image dir and style image dir in ```dataroot```, then run
``` python
python -m experiments CelebA test swapping_grid
```

## Citation
If you find our work helpful for your research, please consider citing our paper.
``` python
@article{wang2025domain,
  title={Domain generalizable portrait style transfer},
  author={Wang, Xinbo and Xu, Wenju and Zhang, Qing and Zheng, Wei-Shi},
  journal={arXiv preprint arXiv:2507.04243},
  year={2025}
}
```
