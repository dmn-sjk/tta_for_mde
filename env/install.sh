#!/bin/bash

conda create --prefix ./ada_depth_shift python=3.8
conda activate ada_depth_shift

python -m pip install -r requirements.txt
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install tensorboardX timm numpy scipy matplotlib pillow tqdm

python -m pip install -r requirements_dgp.txt

python -m pip install waymo-open-dataset-tf-2.11.0==1.5.0
python -m pip install protobuf==3.20
python -m pip install imagecorruptions
python -m pip install openmim
mim install mmcv-full

python -m pip install lpips
