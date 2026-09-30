# Test-Time Adaptation for Monocular Depth Estimation
Code release for the paper
**Adaptive Monocular Depth Estimation with Masked Image Consistency**, D. Sójka, M. Masana, B. Twardowski, and S. Cygert,
2nd Workshop on Test-Time Adaptation: Putting Updates to the Test! (PUT),
International Conference on Machine Learning (ICML), 2025.

This repository builds on the test-time domain adaptation framework and data
pipeline originally released for
[*Test-time Domain Adaptation for Monocular Depth Estimation*](https://github.com/Malefikus/ada-depth)
(ICRA 2023), extending it with additional test-time adaptation methods
(including our masked-image-consistency approach), backbones, datasets, and
augmentation pipelines.

## Prerequisites
The codebase has been developed and tested with python==3.8 and
pytorch==1.13.1; see [`env/install.sh`](env/install.sh) for the exact conda
environment setup (including SHIFT dataset support). To create the
environment, run:

```Shell
bash env/install.sh
```

An earlier, minimal environment (python==3.6.13, pytorch==1.7.1) used for the
original ICRA 2023 source-model baselines is also provided in
[`environment.yml`](environment.yml).

Other core packages: Weights & Biases (`wandb`) and tensorboardX for
experiment logging, timm, mmcv, numpy, scipy, opencv, matplotlib, pillow,
tqdm, PyYAML, json and pickle.

Our code is partially based on [monodepth2](https://github.com/nianticlabs/monodepth2),
[NeWCRFs](https://github.com/aliyun/NeWCRFs), the [DGP](https://github.com/TRI-ML/dgp)
toolkit, the [SHIFT dataset](https://www.vis.xyz/shift/) devkit, and
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2).

## Datasets
[KITTI](https://www.cvlibs.net/datasets/kitti/) dataset is used for training the source models.
We use the (refined) depth prediction set of the KITTI dataset; please download as instructed on their website.

[DDAD](https://github.com/TRI-ML/DDAD) dataset can be used for training the source models (train set)
as well as adaptation from KITTI source models. After downloading the dataset as instructed,
please install the [tool](https://github.com/TRI-ML/dgp) (version 0.1) for data loading.

[Waymo](https://waymo.com/open/) dataset is used for adaptation from source models.
We experiment on the perception dataset v1.2.0; please refer to the [tutorials](https://github.com/waymo-research/waymo-open-dataset)
to install the waymo_open_dataset tool for dataset loading and parsing. Weather- and
time-of-day-based subsets used for adaptation (e.g. sunny/rainy, day/night/dawn-dusk) are
defined in [`splits/waymo_weather`](splits/waymo_weather); select them via `--dataset waymo_all6`,
`--dataset waymo_rainy5`, `--dataset waymo_sunny_day5`, or `--dataset waymo_sunny_night5`.

[SHIFT](https://www.vis.xyz/shift/) is a synthetic driving dataset with continuous and discrete
domain shifts, used for adaptation under controlled shift severity. See
[`datasets/shift_dev`](datasets/shift_dev) for the loading tools and
[`datasets/shift_dev/download.sh`](datasets/shift_dev/download.sh) to download the data
(`--dataset shift` or `--dataset shift_discr1:11`).

[DrivingStereo](https://drivingstereo-dataset.github.io/) can be used for adaptation across
weather conditions (cloudy/foggy/rainy/sunny; `--dataset driving_stereo`).

[KITTI-C](datasets/kitti_c) provides corrupted versions of KITTI (e.g. noise, blur, weather) for
robustness evaluation; see [`datasets/kitti_c/corruptions`](datasets/kitti_c/corruptions) to
generate it (`--dataset kitti_c`).

## Getting Started
### Training the Source Models
Please download one of the trained [SwinTransformer]{} backbones and put it in ./models.
Our experiments are conducted with a 224x224 [Swin-L]{https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth}.

To train the supervised source model on the KITTI dataset, run:

```Shell
python train.py --model_name kitti_sup --width 704 --height 352 --data_path PATH_TO_KITTI
```

on DDAD:

```Shell
python train.py --model_name dgp_sup --dataset dgp --width 640 --height 384 --data_path PATH_TO_ddad.json
```

for self-supervised source model on KITTI, run:

```Shell
python train.py --model_name kitti_unsup --monodepth --width 1216 --height 352 --data_path PATH_TO_KITTI
```

on DDAD:

```Shell
python train.py --model_name dgp_unsup --monodepth --dataset dgp --width 960 --height 608 --data_path PATH_TO_ddad.json
```

Other training options please refer to ./options.py

The trained source models will then be saved in ./exp_logs/MODEL_NAME

Besides the default Swin-L/NewCRF backbone (`--sup_model newcrf`), the code also supports DPT
(`--sup_model dpt`), a U-Net baseline (`--sup_model unet`), and
[Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
(`--sup_model depth_anything_v2_vits` / `_vitb` / `_vitl`) as source models; point
`--sup_model_path` to the corresponding checkpoint.

### Adaptation
Test-time adaptation methods are registered in [`methods/`](methods) and selected via
`--adaptation_method`. This includes our masked-image-consistency method (`consistency`),
as well as several baselines used for comparison in the paper: `cotta` (CoTTA), `svdp` (SVDP),
`mae` (masked-reconstruction TTA), `adadepth`, `contrastive`, `feat_align`, `ssl_naive`,
`ssl_naive_supervised`, `ssl_frozen`, and `sup_frozen`.

To adapt the source models trained on KITTI to the Waymo weather/time-of-day benchmark, run:

```Shell
python adaptation.py \
    --model_name kitti2waymo_consistency \
    --dataset waymo_all6 \
    --data_path PATH_TO_WAYMO \
    --adaptation_method consistency \
    --models_to_load encoder depth \
    --sup_model_path ./exp_logs/kitti_sup/models/weights_19 \
    --ssl_model_path ./exp_logs/kitti_unsup/models/weights_19 \
    --thres 0.4 --learning_rate 1e-5 --num_workers 0
```

To adapt to DDAD, Kitti-C, SHIFT, or DrivingStereo instead, change `--dataset` to `dgp`,
`kitti_c`, `shift` (or `shift_discr1:11`), or `driving_stereo`, respectively, and point
`--data_path` to the corresponding dataset.

Masking and augmentation pipelines for the masked-image-consistency method are controlled via
`--masking` / `--mask_ratio`, and `--augs_config <name>`, which loads one of the YAML pipelines
in [`configs/augmentations`](configs/augmentations) (e.g. `--augs_config cotta_with_geom`).
Use `--loops N` to run repeated online-adaptation passes over a sequence.

The accuracy of the source models as well as the online adaptation will be shown, and logged to
Weights & Biases by default (`--logging_backend wandb`, `--wandb_project`, `--wandb_entity`); pass
`--logging_backend tensorboard` to log to TensorBoard instead.


## Citation
If you find our work useful, please kindly cite:

```
@inproceedings{Sojka_ICMLW2025,
    title={Adaptive Monocular Depth Estimation with Masked Image Consistency},
    author={Damian Sójka and Marc Masana and Bartłomiej Twardowski and Sebastian Cygert},
    booktitle={2nd Workshop on Test-Time Adaptation: Putting Updates to the Test! (PUT), International Conference on Machine Learning (ICML)},
    year={2025}
}
```

This codebase builds on and extends:

```
@inproceedings{Li_ICRA2023,
    title={Test-time Domain Adaptation for Monocular Depth Estimation},
    author={Zhi Li and Shaoshuai Shi and Bernt Schiele and Dengxin Dai},
    booktitle = {International Conference on Robotics and Automation (ICRA)},
    year={2023}
}
```

