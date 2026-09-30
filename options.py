# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        # PATHS
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=None)
        self.parser.add_argument("--depth_path",
                                 type=str,
                                 help="waymo depth path",
                                 default=None)
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default="./exp_logs/")

        # TRAINING options
        self.parser.add_argument("--monodepth",
                                 help="if set train self-supervised (monodepth) model",
                                 action="store_true")
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_zhou_depth", "eigen_full", "odom", "benchmark"],
                                 default="eigen_zhou_depth")
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=50,
                                 choices=[18, 34, 50, 101, 152])
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti_depth",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test", "dgp", "waymo", "waymo_all6", 
                                          "waymo_rainy5", "waymo_sunny_day5", "waymo_sunny_night5",
                                          "kitti_c", "kitti_c_1216x352", "kitti_c_1024x320", "driving_stereo", "shift",
                                          "shift_discr1:11"])
        self.parser.add_argument("--corruption",
                                 type=str,
                                 help="corruption of kitti_c",
                                 default=None,
                                 choices=["impulse_noise", "brightness"])
        self.parser.add_argument("--severity",
                                 type=int,
                                 help="corruption of kitti_c",
                                 default=5)
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=352)
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=704)
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        self.parser.add_argument("--scales",
                                 nargs="+",
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3]
                                 # default=[0]
                                 )
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])
                                #  default=[0, 's'])
        self.parser.add_argument("--autoblur",
                                 help="autoblur from freq aware paper",
                                 action="store_true")
        self.parser.add_argument("--amb_masking",
                                 help="ambiguity masking from freq aware paper",
                                 action="store_true")
        self.parser.add_argument("--skip_layers",
                                 help="use network with skip layers",
                                 action="store_true")
        self.parser.add_argument("--dropout",
                                 help="use network with dropout",
                                 action="store_true")
        self.parser.add_argument("--single_img_experiment",
                                 help="test only on single image (pair of images) all the combinations of layer skipping",
                                 action="store_true")
        self.parser.add_argument("--loss_experiment",
                                 help="",
                                 action="store_true")
        self.parser.add_argument("--adaptation_method",
                                 type=str,
                                 help="corruption of kitti_c",
                                 default=None,
                                #  choices=["adadepth", "ssl_naive", "ssl_frozen", "ssl_naive_supervised"]
                                )
        self.parser.add_argument("--model_type",
                                 type=str,
                                 help="supervised or self-supervised",
                                 default="supervised",
                                 choices=["supervised", "self-supervised"])
        self.parser.add_argument("--seed",
                                 type=int,
                                 help="random seed",
                                 default=1)
        self.parser.add_argument("--gt_transform",
                                 help="use gt transform between frames for loss calculation",
                                 action="store_true")
        self.parser.add_argument("--scale_alignment",
                                 help="use scale alignment to resize images",
                                 action="store_true")
        self.parser.add_argument("--n_bins",
                                 type=int,
                                 help="number of bins for discretizing depth",
                                 default=-1)

        # AUGMENTATION options
        self.parser.add_argument("--augs_config",
                                 type=str,
                                 help="name of a YAML file that defines a custom "
                                      "augmentation pipeline (overrides --augs)",
                                 default=None)
        self.parser.add_argument("--img_ratios",
                                 nargs="+",
                                 type=float,
                                 help="image scale ratios for geometric augmentations "
                                      "(default: 0.5 0.75 1.0 1.25 1.5 1.75 2.0)",
                                 default=None)
        self.parser.add_argument("--flip_augs",
                                 help="enable flip in geometric augmentations",
                                 action="store_true")
        self.parser.add_argument("--masking",
                                 type=str,
                                 help="",
                                 default="simple")
        self.parser.add_argument("--mask_ratio",
                                 type=float,
                                 help="",
                                 default=0.5)

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        self.parser.add_argument("--pos_thr",
                                 type=float,
                                 help="positive threshold",
                                 default=0.05)
        self.parser.add_argument("--neg_thr",
                                 type=float,
                                 help="negative threshold",
                                 default=0.2)
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)
        self.parser.add_argument("--use_others",
                                 help="if set, train pose and predictive mask net when training fully supervised",
                                 action="store_true")
        self.parser.add_argument("--save_vis",
                                 help="if set, save images to the experiment logger",
                                 action="store_true")

        # ABLATION options
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])
        self.parser.add_argument("--resume",
                                 help="if set, resume from load_weights_folder",
                                 action="store_true")
        self.parser.add_argument("--ssl_model_path",
                                 type=str,
                                 help="path to supervised model weights",
                                 default=None)
        self.parser.add_argument("--sup_model",
                                 type=str,
                                 help="supervised mode to load",
                                 default='newcrf')
        self.parser.add_argument("--sup_model_path",
                                 type=str,
                                 help="path to supervised model weights",
                                 default=None)

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each experiment log",
                                 default=250)
        self.parser.add_argument("--logging_backend",
                                 type=str,
                                 help="experiment logging backend",
                                 default="wandb",
                                 choices=["wandb", "tensorboard"])
        self.parser.add_argument("--wandb_project",
                                 type=str,
                                 help="Weights & Biases project name",
                                 default="ada-depth")
        self.parser.add_argument("--wandb_entity",
                                 type=str,
                                 help="Weights & Biases entity name",
                                 default=None)
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen_benchmark",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10", 
                                    "2011_09_26_0096", "2011_09_26_0117", "2011_09_26_0086"],
                                 help="which split to run eval on")
        self.parser.add_argument("--thres",
                                 type=float,
                                 help="using a threshold value to select better pixels for pseudo gt",
                                 default=0.4)
        self.parser.add_argument("--alpha",
                                 type=float,
                                 help="weight",
                                 default=0.0)
        self.parser.add_argument("--calc_scale",
                                 help="if set, calculate scale from labelled set",
                                 action="store_true")

        self.parser.add_argument("--loops",
                                 type=int,
                                 default=1)

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
