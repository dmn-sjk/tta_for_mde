import os
import copy
import time
import math
import json
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import transform
import matplotlib.cm
import matplotlib.pyplot as plt
from PIL import Image
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
import datasets
import datasets.kitti_c
import networks
from layers import *
from utils.utils import *
from utils.experiment_logger import ExperimentLogger
from options import MonodepthOptions
from tqdm import tqdm
from utils.utils import save_tensor_as_image, calculate_batch_image_entropy, calculate_batch_edge_density
from freq_aware_depth import add_autoblured_inputs
from networks import get_supervised_models, get_self_supervised_models
from methods import METHODS
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP

options = MonodepthOptions()
opts = options.parse()

custom_corruptions_seq = [
    'brightness',
    'fog',
    'contrast',
    'defocus_blur',
    'motion_blur',
    'elastic_transform',
    'gaussian_noise',
    'impulse_noise',
    'shot_noise',
    'jpeg_compression'
]



custom_corruptions_seq = [
    'brightness',
    'fog',
    'contrast',
    'defocus_blur',
    'motion_blur',
    'elastic_transform',
    'gaussian_noise',
    'impulse_noise',
    'shot_noise',
    'jpeg_compression'
]


class Adapt:
    def __init__(self, options):
        self.opt = options
        set_seed(self.opt.seed)

        # self.opt.device = torch.device('cpu' if self.opt.no_cuda else f'cuda:{torch.cuda.current_device()}')
        self.opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(self.opt.device)
        self.opt.MIN_DEPTH = 1e-3
        self.opt.MAX_DEPTH = 80


        self.input_height, self.input_width = None, None
        if self.opt.dataset == "dgp":
            self.opt.data_path = os.path.join(self.opt.data_path, "ddad.json")
            # self.gt_height, self.gt_width = 1920, 1152
            self.cam_h = datasets.DGPDataset.CAM_H
            self.ratio = self.cam_h / datasets.KITTIDataset.CAM_H
            self.opt.height, self.opt.width = 384, 640 # self.height_sec, self.width_sec
        elif "waymo" in self.opt.dataset:
            self.opt.depth_path = os.path.join(self.opt.data_path, 'perception_v_1_2_0/validation')
            self.opt.data_path = os.path.join(self.opt.data_path, 'perception_v_1_2_0/validation')
            # self.gt_height, self.gt_width = 1920, 1248
            self.cam_h = datasets.WaymoDataset.CAM_H
            self.ratio = self.cam_h / datasets.KITTIDataset.CAM_H
            self.opt.height, self.opt.width = 320, 512 # self.height_sec, self.width_sec
        elif self.opt.dataset == "driving_stereo":
            # self.opt.data_path = os.path.join(self.opt.data_path, 'driving_sterieo')
            # self.gt_height, self.gt_width = 352, 1216
            self.input_height, self.input_width = 352, 1216
            self.opt.height, self.opt.width = 352, 1216
            self.ratio = 1
            self.cam_h = datasets.DrivingStereo.CAM_H
            self.ratio = self.cam_h / datasets.KITTIDataset.CAM_H
        elif "shift" in self.opt.dataset:
            self.opt.height, self.opt.width = 896, 1440 # self.height_sec, self.width_sec
            self.cam_h = datasets.SHIFTDataset.CAM_H
            self.ratio = self.cam_h / datasets.KITTIDataset.CAM_H
        else:
            # KITTI, KITTI-C
            # self.gt_height, self.gt_width = 352, 1216
            self.input_height, self.input_width = 352, 1216
            self.opt.height, self.opt.width = 352, 1216
            self.ratio = 1
            self.cam_h = datasets.KITTIDataset.CAM_H

        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name, f'seed{self.opt.seed}')

        # create a new log for every save
        self.writer = ExperimentLogger.from_options(
            self.opt,
            self.log_path,
            run_name=self.opt.seed,
        )

        self.num_scales = len(self.opt.scales)
        self.opt.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else len(self.opt.frame_ids)

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        # TODO: write it better, keep in mind the SSL model
        if self.opt.sup_model == 'dpt':
            self.opt.mean = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1).to(self.opt.device)
            self.opt.std = torch.tensor([0.5, 0.5, 0.5]).view(3,1,1).to(self.opt.device)
        else:
            self.opt.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1).to(self.opt.device)
            self.opt.std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1).to(self.opt.device)

        self.ssim = SSIM()
        self.ssim.to(self.opt.device)

        self.writer.log_config(self.opt)

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and experiment logs are saved to:\n  ", self.opt.log_dir)
        print("Logging backend:\n  ", self.opt.logging_backend)
        print("Training is using:\n  ", self.opt.device)

        # dataloader, test set
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset,
                         "kitti_depth": datasets.KITTIDepthDataset,
                         "dgp": datasets.DGPDataset,
                         "waymo": datasets.WaymoDataset,
                         "waymo_all6": datasets.WaymoDataset,
                         "waymo_rainy5": datasets.WaymoDataset,
                         "waymo_sunny_day5": datasets.WaymoDataset,
                         "waymo_sunny_night5": datasets.WaymoDataset,
                         "kitti_c": datasets.kitti_c.KITTIRAWDataset,
                         'driving_stereo': datasets.DrivingStereo,
                         'shift': datasets.SHIFTDataset,
                         'shift_discr1:11': datasets.SHIFTDataset}
        self.dataset_instance = datasets_dict[self.opt.dataset]
        self.seq_num = 1
                
        # replay data
        # fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")
        # train_filenames = readlines(fpath.format("train"))
        # dataset = datasets.KITTIDepthDataset(self.opt.data_path, train_filenames,
        #                                 352, 1216,
        #                                 [0], # [0, -1, 1], 
        #                                 1,
        #                                 opt=self.opt,
        #                                 is_train=True, # True, # TODO: change to False for driving_stereo, verify what is the best option
        #                                 rotate_aug=False, 
        #                                 pseu=True,
        #                                 depth_path=self.opt.depth_path,
        #                                 img_ext='.png' if self.opt.png else '.jpg',
        #                                 on_the_fly=False
        #                                 )
        # self.replay_dataloader = DataLoader(dataset, 3, shuffle=True,
        #                             num_workers=self.opt.num_workers,
        #                             pin_memory=True, drop_last=False)

        # waymo time of day and weathers
        daytimes = ['Day', 'Dawn/Dusk', 'Night']
        weathers = ['sunny', 'rain']

        # self.weath = weathers[0]
        # self.day = daytimes[0]
        
        self.corruptions = [self.opt.corruption]

        if self.opt.dataset == "dgp":
            filenames = "val"
        elif "waymo" in self.opt.dataset:
            filenames, self.waymo_conditions = datasets.waymo_dataset.get_filenames_and_conditions(self.opt.dataset, self.opt.data_path)
            self.seq_num = len(filenames)
            self.corruptions = [self.opt.corruption for _ in range(self.seq_num)]
        elif self.opt.dataset == "driving_stereo":
            splits_dir = './splits/driving_stereo'
            # TODO: unify the waymo, driving_stereo and kitti_c domains
            domain_sequence = ['foggy', 'rainy', 'sunny', 'cloudy']
            filenames = []
            for domain in domain_sequence:
                filenames.append(readlines(os.path.join(splits_dir, domain, "test_files.txt")))
                for i in range(len(filenames[-1])):
                    filenames[-1][i] = os.path.join(domain, domain, 'left-image-full-size', filenames[-1][i])
            self.seq_num = len(filenames)
            self.corruptions = [self.opt.corruption for _ in range(self.seq_num)]
        else:
            splits_dir = './splits/'
            filenames = readlines(os.path.join(splits_dir, self.opt.eval_split, "val_files_bak.txt"))

            if self.opt.dataset == 'kitti_c':
                self.corruptions = []
                # for corruption in corruption_dict.keys():
                for corruption in custom_corruptions_seq:
                    self.corruptions.append(corruption)
                self.seq_num = len(self.corruptions)
        
        self.filenames = filenames

        self.adaptation_method = METHODS[self.opt.adaptation_method](self.opt,
                                                                 gen_imgs_pred_func=self.generate_images_pred,
                                                                 predict_poses_func=self.predict_poses,
                                                                 compute_losses_unsup_func=self.compute_losses_unsup,
                                                                 adapt_instance=self)

    def run_adapt(self):
        self.start_time = time.time()
        self.step = 0

        errors_tt = {}
        errors_t = {}
        
        repeats = self.opt.loops
        for rep in range(repeats):
            errors_loop = {}
            for i in range(self.seq_num):
                # self.process_batch = METHODS[self.opt.adaptation_method](self.opt,
                #                                                         gen_imgs_pred_func=self.generate_images_pred,
                #                                                         predict_poses_func=self.predict_poses,
                #                                                         compute_losses_unsup_func=self.compute_losses_unsup,
                #                                                         adapt_instance=self).process_batch
                print("seq {}".format(i))
                if "waymo" in self.opt.dataset:
                    filenames = [self.filenames[i]]
                elif self.opt.dataset == "driving_stereo":
                    filenames = self.filenames[i]
                else:
                    filenames = self.filenames
                dataset = self.dataset_instance(self.opt.data_path, filenames,
                                                self.input_height, self.input_width,
                                                self.opt.frame_ids, # [0, -1, 1], 
                                                self.num_scales,
                                                opt=self.opt,
                                                is_train=True, # True, # TODO: change to False for driving_stereo, verify what is the best option
                                                rotate_aug=False, 
                                                pseu=True,
                                                depth_path=self.opt.depth_path,
                                                img_ext='.png' if self.opt.png else '.jpg',
                                                eval_corr_type=(self.corruptions[i], self.opt.severity),
                                                on_the_fly=False, gt_transformation=self.opt.gt_transform)

                self.dataloader = DataLoader(dataset, 1, shuffle=False,
                                            num_workers=self.opt.num_workers,
                                            pin_memory=True, drop_last=False)

                pbar = tqdm(total = len(self.dataloader))
                errors = {}

                dataset = self.opt.dataset
                if "waymo" in self.opt.dataset:
                    dataset += '_' + self.waymo_conditions[i]
                self.dataset = dataset
                
                if self.opt.loss_experiment:
                    save_dir = os.path.join('playground_new', 'single_stereo_experiment_gtT', dataset)
                    num_batches = 12
                    every_n_batches = len(self.dataloader) // num_batches
                    chosen_batch_idxs = [i * every_n_batches for i in range(num_batches)]
                    num_predictions = 1000
                
                # save_dir = os.path.join('dataset_imgs', 'kitti_2011_09_26_0096')
                # os.makedirs(save_dir, exist_ok=True)
                torch.cuda.reset_peak_memory_stats()
                for batch_idx, inputs in enumerate(self.dataloader):
                    # save_tensor_as_image(inputs[('color_uncrop', 0, 0)], os.path.join('test', f'{batch_idx}.png'))
                    # save_tensor_as_image(inputs[('color_uncrop', 's', 0)], os.path.join('test', f'{batch_idx}_s.png'))
                    
                    # Get next batch from self.replay_dataloader
                    # try:
                    #     replay_inputs = next(self.replay_dataloader_iter)
                    # except (StopIteration, AttributeError):
                    #     # Initialize or reset the iterator if it doesn't exist or is exhausted
                    #     self.replay_dataloader_iter = iter(self.replay_dataloader)
                    #     replay_inputs = next(self.replay_dataloader_iter)
                    # inputs['replay'] = replay_inputs[('color_uncrop', 0, 0)]
                    # inputs['replay_gt_depth'] = replay_inputs['depth_gt_uncrop']
                    # print(inputs['replay'].shape)
                    # print(inputs['replay_gt_depth'].shape)
                    

                    if self.opt.loss_experiment:
                        if batch_idx not in chosen_batch_idxs:
                            continue
                    self.backproject_depth = None
                    self.project_3d = None
                    self.scale_recovery = None
                    torch.cuda.empty_cache()

                    self.height = inputs["height_sec"].to(self.opt.device)
                    self.width = inputs["width_sec"].to(self.opt.device)
                    self.scale_recovery = ScaleRecovery(1, self.height, self.width).to(self.opt.device)

                    # checking height and width are multiples of 32
                    assert self.height % self.opt.patch_size == 0, "'height' must be a multiple of patch_size"
                    assert self.width % self.opt.patch_size == 0, "'width' mustbe a multiple of patch_size"

                    self.backproject_depth = {}
                    self.project_3d = {}
                    for scale in self.opt.scales:
                        h = self.height // (2 ** scale)
                        w = self.width // (2 ** scale)

                        self.backproject_depth[scale] = BackprojectDepth(1, h, w)
                        self.backproject_depth[scale].to(self.opt.device)

                        self.project_3d[scale] = Project3D(1, h, w)
                        self.project_3d[scale].to(self.opt.device)

                    self.step += 1
                    before_op_time = time.time()
                    
                    if self.opt.autoblur:
                        add_autoblured_inputs(self.opt.scales, self.opt.frame_ids, inputs)


                    if self.opt.single_img_experiment:
                        data = {
                            'unsup_loss': [],
                            'rmse': []
                        }
                        reg_outputs_list = []
                        
                        for key, ipt in inputs.items():
                            inputs[key] = ipt.to(self.opt.device)

                        # folder = f"playground/waymo/first_images_{self.weath}_{self.day.replace('/', '_')}"
                        # folder = f"playground/kitti_c/first_images_{self.opt.corruption}{self.opt.severity}"
                        # os.makedirs(folder, exist_ok=True)
                        # save_tensor_as_image(inputs[('color_uncrop', -1, 0)], os.path.join(folder, '0.png'))
                        # save_tensor_as_image(inputs[('color_uncrop', 0, 0)], os.path.join(folder, '1.png'))
        
                        for m in self.reg_models.values():
                            m.eval()
                        with torch.no_grad():
                            predicted_poses = self.predict_poses(inputs, None, self.reg_models)

        
                        for i in range(self.reg_models["encoder"].module.encoder.num_of_mask_combinations):
                            print(i)
                            
                            # change skip mask
                            # self.reg_models["encoder"].module.encoder.set_mask_combination[self.reg_models["encoder"].module.encoder.num_of_mask_combinations - i - 1]


                            # error, error_local, error_unsup, outputs, losses,\
                            # error_ref, error_local_ref, error_unsup_ref, error_teacher,\
                            # error_teacher_local = self.adaptation_method.process_batch(inputs)
                            
                            
                            error_unsup, losses, reg_outputs = self.process_batch_save_unsup_preds(inputs, predicted_poses, reg_outputs_list)
                            
                            rmse = error_unsup[2]
                            unsup_loss = losses['unsup_loss']
                            data['unsup_loss'].append(unsup_loss)
                            data['rmse'].append(rmse)
                            
                            save = False
                            if i % 500 == 0:
                                save = True
                            
                            # i, error_unsup (list of errors), unsup_losses['loss']
                            # if save:
                            #     to_save = {}
                            #     for key, val in data.items():
                            #         if isinstance(val, torch.Tensor):
                            #             to_save[key] = np.array(val.item())
                            #         else:
                            #             to_save[key] = np.array(val)
                                        
                            #     # np.save(f"playground/waymo/{self.weath}_{self.day.replace('/', '_')}_skip_loss_err_single_image_2frames.npy", 
                            #             # to_save)
                            #     np.save(f"playground/kitti_c/{self.opt.corruption}{self.opt.severity}_skip_loss_err_single_image_2frames_.npy", 
                            #             to_save)
                                # torch.save(reg_outputs_list, 'playground/dgp/unsup_outputs_single_image_dgp_2frames_every16th.pth')

                            torch.cuda.empty_cache()
                    
                    elif self.opt.loss_experiment:

                        data = {
                            'unsup_loss': [],
                            'unsup_loss_wo_identity': [],
                            'unsup_loss_wo_identity_&_smooth': [],
                            'rmse': [],
                            'img_entropy': [],
                            'img_density': [],
                            'identity_selection': []
                        }
                        reg_outputs_list = []

                        inputs_to_device(inputs, self.opt.device)

                        # folder = f"playground/waymo/first_images_{self.weath}_{self.day.replace('/', '_')}"
                        # folder = f"playground/kitti_c/first_images_{self.opt.corruption}{self.opt.severity}"
                        folder = os.path.join(save_dir, f"batch_idx_{batch_idx}")
                        imgs_folder = os.path.join(folder, 'imgs')
                        os.makedirs(imgs_folder, exist_ok=True)
                        preds_folder = os.path.join(folder, 'preds')
                        os.makedirs(preds_folder, exist_ok=True)
                        
                        
                        for k in self.opt.frame_ids:
                            save_tensor_as_image(inputs[('color_uncrop', k, 0)], os.path.join(imgs_folder, f'{k}.png'))
                        
                        # entropies = []
                        # densities = []
                        # for i in range(2):
                        #     img_entropy = calculate_batch_image_entropy(inputs["color_uncrop", 0 - i, 0].cpu())
                        #     img_edge_density = calculate_batch_edge_density(inputs["color_uncrop", 0 - i, 0].cpu())
                            
                        #     if isinstance(img_entropy, torch.Tensor):
                        #         img_entropy = img_entropy.item()
                        #     if isinstance(img_edge_density, torch.Tensor):
                        #         img_edge_density = img_edge_density.item()
                            
                        #     entropies.append(img_entropy)
                        #     densities.append(img_edge_density)
                        
                        # data['img_entropy'].append(entropies)
                        # data['img_density'].append(densities)
        
        
                        for m in self.reg_models.values():
                            m.eval()
                        with torch.no_grad():
                            predicted_poses = self.predict_poses(inputs, None, self.reg_models)

        
                        for i in range(num_predictions):
                            # print(i)
                            if i % 100 == 0:
                                save_pred_path = os.path.join(preds_folder, str(i) + '.png')
                            else:
                                save_pred_path = None
                            
                            error_unsup, losses, reg_outputs = self.process_batch_save_unsup_preds(inputs, predicted_poses, reg_outputs_list, 
                                                                                                save_pred_path=save_pred_path)
                            
                            rmse = error_unsup[2]
                            unsup_loss = losses['unsup_loss']
                            unsup_loss_wo_identity = losses['unsup_loss_wo_identity']
                            unsup_loss_wo_identity_smooth = losses['unsup_loss_wo_identity_&_smooth']
                            if isinstance(unsup_loss, torch.Tensor):
                                unsup_loss = unsup_loss.item()
                            if isinstance(unsup_loss_wo_identity, torch.Tensor):
                                unsup_loss_wo_identity = unsup_loss_wo_identity.item()
                            if isinstance(unsup_loss_wo_identity_smooth, torch.Tensor):
                                unsup_loss_wo_identity_smooth = unsup_loss_wo_identity_smooth.item()
                            if isinstance(rmse, torch.Tensor):
                                rmse = rmse.item()
                            data['unsup_loss'].append(unsup_loss)
                            data['unsup_loss_wo_identity'].append(unsup_loss_wo_identity)
                            data['unsup_loss_wo_identity_&_smooth'].append(unsup_loss_wo_identity_smooth)
                            data['rmse'].append(rmse)
                            
                            identity_selection = (reg_outputs["identity_selection/0"].sum() / torch.numel(reg_outputs["identity_selection/0"])).cpu().item()                        
                            data['identity_selection'].append(identity_selection)

                            torch.cuda.empty_cache()
                            
                        save_data_dict(data, os.path.join(folder, 'data.npy'))
                        continue

                    else:
                        # error, error_local, error_unsup, outputs, losses,\
                        # error_ref, error_local_ref, error_unsup_ref, error_teacher,\
                        # error_teacher_local = self.adaptation_method.process_batch(inputs)
                        outputs, metrics, losses = self.adaptation_method.process_batch(inputs)

                    for key, val in metrics.items():
                        errors.setdefault(key, []).append(val)
                        errors_tt.setdefault(key, []).append(val)
                        errors_loop.setdefault(key, []).append(val)

                
                    if batch_idx % 1 == 0:
                        mean_errors_dict = {}
                        
                        for key, val in errors_tt.items():
                            mean_errors_100 = np.array(val).mean(0)
                            if 'local' in  key:
                                metric_names = DEPTH_METRIC_NAMES_LOCAL
                            elif 'ref' in key:
                                metric_names = DEPTH_METRIC_NAMES_UNSUP
                            else:
                                metric_names = DEPTH_METRIC_NAMES
                                
                            for i, metric in enumerate(metric_names):
                                mean_errors_dict[key + '/' + metric] = mean_errors_100[i]
                        
                        for key, val in metrics.items():
                            # NOTE: each metric have to have len() equal to self.depth_metric_names len()
                            if 'local' in  key:
                                metric_names = DEPTH_METRIC_NAMES_LOCAL
                            elif 'ref' in key:
                                metric_names = DEPTH_METRIC_NAMES_UNSUP
                            else:
                                metric_names = DEPTH_METRIC_NAMES
                            for i, metric in enumerate(metric_names):
                                mean_errors_dict[key + '/not_mean_' + metric] = val[i]
                        
                        if self.opt.save_vis and batch_idx % 100 == 0:
                            save_imgs = True
                        else:
                            save_imgs = False
                        
                        self.log(inputs, outputs, mean_errors_dict, losses, save_imgs)

                    pbar.update(1)
                    # pbar.set_description("abs_rel diff: {:.4f}, abs_rel ref: {:.4f}".format(error[0]-error_ref[0], error_ref[0]))
                    pbar.set_description("error: {:.4f}".format(metrics['error'][0]))
                    
                    del outputs
                    del metrics
                    del losses
                    torch.cuda.empty_cache()
                
                if self.opt.loss_experiment:
                    continue

                mean_errors = {}
                for key, val in errors.items():
                    mean_errors[key] = np.array(val).mean(0)
                    
                # pbar.close()

                # print("\n  " + ("{:>8} | " * 8).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "median"))
                # print("supervised w/o gt median scaling (inital/teacher/student)")
                # if mean_errors_ref is not None:
                #     print(("&{: 8.3f}  " * 8).format(*mean_errors_ref.tolist()) + "\\\\")
                # if mean_errors_teacher is not None:
                #     print(("&{: 8.3f}  " * 8).format(*mean_errors_teacher.tolist()) + "\\\\")
                # if mean_errors is not None:
                #     print(("&{: 8.3f}  " * 8).format(*mean_errors.tolist()) + "\\\\")
                # print("self-supervised w gt median scaling (inital/adapted)")
                # if mean_errors_unsup_ref is not None:
                #     print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup_ref.tolist()) + "\\\\")
                # if mean_errors_unsup is not None:
                #     print(("&{: 8.3f}  " * 8).format(*mean_errors_unsup.tolist()) + "\\\\")

                for key, val in mean_errors.items():
                    errors_t.setdefault(key, []).append(val)
        
            if self.opt.loss_experiment:
                import sys
                sys.exit()

            self.adaptation_method.avg_depth_errors.log_charts(self.writer, self.step)
            self.adaptation_method.avg_depth_errors_median_scaling.log_charts(self.writer, self.step)

            mean_errors_loop = {}
            for key, val in errors_loop.items():
                mean_errors_loop[key] = np.array(val).mean(0)

            loop_avg_metrics = {}
            for key, val in mean_errors_loop.items():
                if 'local' in key:
                    metric_names = DEPTH_METRIC_NAMES_LOCAL
                elif 'ref' in key:
                    metric_names = DEPTH_METRIC_NAMES_UNSUP
                else:
                    metric_names = DEPTH_METRIC_NAMES
                for i, metric in enumerate(metric_names):
                    loop_avg_metrics["loop_avg_per_rep/{}".format(key + '/' + metric)] = val[i]

            self.log_loop_metrics(loop_avg_metrics)

            mean_errors_t = {}
            for key, val in errors_t.items():
                mean_errors_t[key] = np.array(val).mean(0)

            metric_names = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3", "median"]

            def _errors_to_dict(errors_array):
                return {name: round(float(val), 3) for name, val in zip(metric_names, errors_array.tolist())}

            summary_json = {}

            category_keys = {
                "supervised_wo_gt_median_scaling": ["error_ref", "error_teacher", "error"],
                "supervised_w_gt_median_scaling": ["error_local_ref", "error_teacher_local", "error_local"],
                "self_supervised_w_gt_median_scaling": ["error_unsup_ref", "error_unsup"],
            }

            for category, keys in category_keys.items():
                for k in keys:
                    if k in mean_errors_t:
                        summary_json.setdefault(category, {})[k] = _errors_to_dict(mean_errors_t[k])
                        del mean_errors_t[k]

            if len(mean_errors_t):
                for k in list(mean_errors_t.keys()):
                    summary_json.setdefault("other", {})[k] = _errors_to_dict(mean_errors_t[k])

            # Print human-readable summary to console
            summary_log = ''
            summary_log += "total\n"
            summary_log += "\n  " + ("{:>8} | " * 8).format(*metric_names) + '\n'
            for category, entries in summary_json.items():
                summary_log += f"{category}\n"
                for k, metrics in entries.items():
                    summary_log += f"  {k}: " + ("  {: 8.3f}" * 8).format(*metrics.values()) + "\n"
            summary_log += "\n-> Done!\n"
            print(summary_log)

            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            if repeats > 1:
                with open(os.path.join(self.log_path, f'rep{rep}_summary.json'), 'w') as f:
                    json.dump(summary_json, f, indent=2)
                self.writer.add_summary(summary_json, "final/rep_{}".format(rep), self.step)
            else:
                with open(os.path.join(self.log_path, 'summary.json'), 'w') as f:
                    json.dump(summary_json, f, indent=2)
                self.writer.add_summary(summary_json, "final", self.step)
        
        log_parent_dir = os.path.dirname(self.log_path) # where seed folder reside

        # Average summary.json results across all seed folders
        seed_dirs = sorted([
            d for d in os.listdir(log_parent_dir)
            if os.path.isdir(os.path.join(log_parent_dir, d)) and d.startswith('seed')
        ])
        seed_summaries = []
        for sd in seed_dirs:
            summary_path = os.path.join(log_parent_dir, sd, 'summary.json')
            if os.path.isfile(summary_path):
                with open(summary_path, 'r') as f:
                    seed_summaries.append(json.load(f))

        if len(seed_summaries) > 1:
            avg_summary = {}
            std_summary = {}
            for category in seed_summaries[0]:
                avg_summary[category] = {}
                std_summary[category] = {}
                for key in seed_summaries[0][category]:
                    avg_summary[category][key] = {}
                    std_summary[category][key] = {}
                    for metric in seed_summaries[0][category][key]:
                        vals = [
                            s[category][key][metric]
                            for s in seed_summaries
                            if category in s and key in s[category] and metric in s[category][key]
                        ]
                        avg_summary[category][key][metric] = round(float(np.mean(vals)), 3)
                        std_summary[category][key][metric] = round(float(np.std(vals)), 3)

            all_seeds_summary = {
                "num_seeds": len(seed_summaries),
                "seed_dirs": seed_dirs,
                "mean": avg_summary,
                "std": std_summary,
            }
            with open(os.path.join(log_parent_dir, 'summary.json'), 'w') as f:
                json.dump(all_seeds_summary, f, indent=2)
            self.writer.add_summary(avg_summary, "final_all_seeds/mean", self.step)
            self.writer.add_summary(std_summary, "final_all_seeds/std", self.step)
            print(f"Saved averaged summary over {len(seed_summaries)} seeds to {log_parent_dir}/summary.json")

    def process_batch_save_unsup_preds(self, inputs, predicted_poses, reg_outputs_list, save_pred_path=None):
        with torch.no_grad():
            reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            reg_outputs = self.reg_models["depth"](reg_features)
            reg_outputs.update(predicted_poses)
            pred_disp, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            
            # disp = depth_to_disp(inputs['depth_gt_uncrop'], self.opt.min_depth, self.opt.max_depth)
            # _, reg_depth_unsup = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            # reg_outputs[("disp", 0)] = disp
            
            self.generate_images_pred(inputs, reg_outputs)
            unsup_losses = self.compute_losses_unsup(inputs, reg_outputs)
        
        # self.match_orb_features(inputs["color_uncrop", 0, 0].cpu(), inputs["color_uncrop", -1, 0].cpu(), reg_outputs[("sample", -1, 0)])
        
        error_unsup = list(self.compute_depth_errors(inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=True))

        for idx, term in enumerate(error_unsup):
            error_unsup[idx] = term.detach().cpu().numpy()

        losses = {}
        for loss_type, loss_val in unsup_losses.items():
            losses['unsup_' + loss_type] = loss_val
   
        # save images
        # dgp
        # if losses['unsup_loss'] > 0.2025 and losses['unsup_loss'] < 0.2035:
        # kitti
        # if losses['unsup_loss'] > 0.128 and losses['unsup_loss'] < 0.132:
        # waymo
        # if losses['unsup_loss'] > 0.0795 and losses['unsup_loss'] < 0.0805:
            # loss_folder = os.path.join(f"playground/waymo/{self.weath}_{self.day.replace('/', '_')}_pred_visualization", 
            #                            f"{losses['unsup_loss']:.5f}")
        # if True:
        #     loss_folder = os.path.join(f"playground/kitti_c/{self.opt.corruption}{self.opt.severity}_pred_visualization", 
        #                                f"{losses['unsup_loss']:.5f}")
        #     rmse = error_unsup[2]
        #     rmse_folder = os.path.join(loss_folder, str(rmse))
        #     os.makedirs(rmse_folder, exist_ok=True)
            
        #     save_tensor_as_image(unsup_losses['loss_map/0'] / unsup_losses['loss_map/0'].max() , os.path.join(rmse_folder, f'loss_map_scale0_{rmse}.png'))
        #     save_tensor_as_image(reg_outputs[('color', -1, 0)], os.path.join(rmse_folder, f'reprojected_scale0_{rmse}.png'))
            # reg_depth_unsup = reg_depth_unsup.cpu()[:, 0]
            # save_tensor_as_image(reg_depth_unsup / reg_depth_unsup.max(), os.path.join(rmse_folder, f'depth_scale0_{rmse}.png'))
            
        #     # save_tensor_as_image(inputs['depth_gt_uncrop'] / inputs['depth_gt_uncrop'].max(), os.path.join(rmse_folder, f'gt_depth_scale0_{rmse}.png'))
        #     save_tensor_as_image(inputs['depth_gt_uncrop'] / (inputs['depth_gt_uncrop'].max() * 0.7), 
        #                          os.path.join(rmse_folder, f'gt_depth_scale0_{rmse}.png'))
            
        if save_pred_path is not None:
            reg_depth_unsup = reg_depth_unsup.cpu()[:, 0]
            save_tensor_as_image(reg_depth_unsup / reg_depth_unsup.max(), os.path.join(save_pred_path))

        return error_unsup, losses, reg_outputs
    

    def predict_poses(self, inputs, features, models):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}

        if self.opt.gt_transform:
            for f_i in self.opt.frame_ids[1:]:
                outputs[("cam_T_cam", 0, f_i)] = inputs[("gt_cam_T_cam", 0, f_i)]
        elif self.opt.model_type == "supervised":
            # ORB-based ego-motion estimation
            for f_i in self.opt.frame_ids[1:]:
                if f_i == 's':
                    raise NotImplementedError("Stereo not implemented yet")
                
                # For monocular sequences, estimate pose using ORB features
                current_frame = inputs["color_uncrop", 0, 0].cpu().numpy()[0]  # Current frame (t)
                prev_frame = inputs["color_uncrop", f_i, 0].cpu().numpy()[0]   # Previous frame (t-1)
                
                # Convert to uint8 format for OpenCV
                current_frame = (current_frame * 255).astype(np.uint8)
                prev_frame = (prev_frame * 255).astype(np.uint8)
                current_frame = np.transpose(current_frame, (1, 2, 0))
                prev_frame = np.transpose(prev_frame, (1, 2, 0))
                
                # Convert to grayscale
                current_gray = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
                
                # Initialize ORB detector
                orb = cv2.ORB_create(
                    nfeatures=2000,
                    scaleFactor=1.2,
                    nlevels=8,
                    edgeThreshold=31,
                    firstLevel=0,
                    WTA_K=2,
                    patchSize=31,
                    fastThreshold=20
                )
                
                # Detect keypoints and compute descriptors
                kp1, des1 = orb.detectAndCompute(prev_gray, None)
                kp2, des2 = orb.detectAndCompute(current_gray, None)
                
                if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
                    # Match features using FLANN matcher
                    FLANN_INDEX_LSH = 6
                    index_params = dict(algorithm=FLANN_INDEX_LSH, 
                                        table_number=6, 
                                        key_size=12, 
                                        multi_probe_level=1)
                    search_params = dict(checks=50)
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    
                    matches = flann.knnMatch(des1, des2, k=2)
                    
                    # Apply Lowe's ratio test
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:
                                good_matches.append(m)
                    
                    if len(good_matches) > 10:
                        # Extract matched keypoints
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        
                        # Get camera intrinsics
                        K = inputs[("K", 0)].cpu().numpy()[0, :3, :3]

                        # Estimate essential matrix using RANSAC
                        E, mask = cv2.findEssentialMat(
                            src_pts, dst_pts, K, 
                            method=cv2.RANSAC, 
                            prob=0.999, 
                            threshold=1.0
                        )
                        
                        if E is not None:
                            # Recover pose from essential matrix
                            _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)
                            
                            # Get scale from ground truth transform if available
                            if ("gt_cam_T_cam", 0, f_i) in inputs:
                                gt_T = inputs[("gt_cam_T_cam", 0, f_i)].cpu().numpy()
                                gt_translation = gt_T[:3, 3]
                                gt_translation_norm = np.linalg.norm(gt_translation)
                                
                                # Scale the ORB translation to match GT scale
                                if gt_translation_norm > 0:
                                    orb_translation_norm = np.linalg.norm(t)
                                    if orb_translation_norm > 0:
                                        scale_factor = gt_translation_norm / orb_translation_norm
                                        t_scaled = t * scale_factor
                                    else:
                                        t_scaled = t
                                else:
                                    t_scaled = t
                            else:
                                t_scaled = t
                            
                            # Convert to transformation matrix
                            T = np.eye(4)
                            T[:3, :3] = R
                            T[:3, 3] = t_scaled.flatten()
                            
                            # Convert to tensor and add to outputs
                            T_tensor = torch.from_numpy(T).float().to(self.opt.device)
                            outputs[("cam_T_cam", 0, f_i)] = T_tensor
                        else:
                            print("pose estimation failed")
                            # Fallback: identity transformation if pose estimation fails
                            T_tensor = torch.eye(4).float().to(self.opt.device)
                            outputs[("cam_T_cam", 0, f_i)] = T_tensor
                    else:
                        print("not enough matches")
                        # Fallback: identity transformation if not enough matches
                        T_tensor = torch.eye(4).float().to(self.opt.device)
                        outputs[("cam_T_cam", 0, f_i)] = T_tensor
                else:
                    print("no features detected")
                    # Fallback: identity transformation if no features detected
                    T_tensor = torch.eye(4).float().to(self.opt.device)
                    outputs[("cam_T_cam", 0, f_i)] = T_tensor
        else:
            # select what features the pose network takes as input
            pose_feats = {f_i: (inputs["color_uncrop", f_i, 0]-self.opt.mean)/self.opt.std for f_i in self.opt.frame_ids}

            for f_i in self.opt.frame_ids[1:]:
                # To maintain ordering we always pass frames in temporal order
                if f_i != 's' and f_i < 0:
                    pose_inputs = [pose_feats[f_i], pose_feats[0]]
                else:
                    pose_inputs = [pose_feats[0], pose_feats[f_i]]

                pose_inputs = [models["pose_encoder"](torch.cat(pose_inputs, 1))]

                axisangle, translation = models["pose"](pose_inputs)
                outputs[("axisangle", 0, f_i)] = axisangle
                outputs[("translation", 0, f_i)] = translation

                # Invert the matrix if the frame id is negative
                outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                    axisangle[:, 0], translation[:, 0], invert=(f_i != 's' and f_i < 0))
 
        return outputs


    def generate_images_pred(self, inputs, outputs, disp_input=True, source_im_key='color_uncrop'):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        if self.opt.autoblur:
            source_im_key = 'color_uncrop_autoblur'
        else:
            source_im_key = 'color_uncrop'
            
        for scale in self.opt.scales:
            # disp = outputs[("disp", scale)]
            # disp = F.interpolate(
            #     disp, [self.height, self.width], mode="bilinear", align_corners=False)
            # _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            if disp_input:
                disp = outputs[("disp", scale)]
                disp = F.interpolate(
                    disp, [self.height, self.width], mode="bilinear", align_corners=False)
                _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)
            else:
                depth = outputs[('depth', scale)]
                depth = F.interpolate(
                    depth, [self.height, self.width], mode="bilinear", align_corners=False)

            source_scale = 0
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):
                if frame_id == 's':
                    T = inputs['stereo_T']
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]
                    
                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)

                outputs[("sample", frame_id, scale)] = pix_coords

                source_image = inputs[(source_im_key, frame_id, source_scale)]

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    source_image,
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                outputs[("color_identity", frame_id, scale)] = source_image


    def compute_reprojection_loss(self, pred, target):
        """Computes reprojection loss between a batch of predicted and target images
        """
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)

        ssim_loss = self.ssim(pred, target).mean(1, True)
        reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

        return reprojection_loss


    def compute_losses_unsup(self, inputs, outputs, disp_input=True, mask=None):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0
        total_loss_wo = 0
        total_loss_wo_smooth = 0

        for scale in self.opt.scales:
            loss = 0
            reprojection_losses = []
            source_scale = 0

            if disp_input:
                disp = outputs[("disp", scale)]
            else:
                depth = outputs[('depth', scale)]
                disp = depth_to_disp(depth, self.opt.min_depth, self.opt.max_depth)
                
            color = inputs[("color_uncrop_autoblur", 0, scale)] if not self.opt.amb_masking and self.opt.autoblur \
                else inputs[("color_uncrop", 0, scale)]
            if self.opt.autoblur:
                target = inputs[("color_uncrop_autoblur", 0, source_scale)]
            else:
                target = inputs[("color_uncrop", 0, source_scale)]

            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            identity_reprojection_losses = []
            for frame_id in self.opt.frame_ids[1:]:
                if self.opt.autoblur:
                    pred = inputs[("color_uncrop_autoblur", frame_id, source_scale)]
                else:
                    pred = inputs[("color_uncrop", frame_id, source_scale)]
                identity_reprojection_losses.append(
                    self.compute_reprojection_loss(pred, target))
            identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
            identity_reprojection_loss = identity_reprojection_losses
            reprojection_loss = reprojection_losses

            if self.opt.amb_masking:
                # ambiguity mask
                ambiguity_mask = self.compute_ambiguity_mask( # DIFFERENCE
                    inputs, outputs, reprojection_loss, scale)
                if mask is not None:
                    mask = torch.logical_and(ambiguity_mask, mask)
                else:
                    mask = ambiguity_mask

            # add random numbers to break ties
            identity_reprojection_loss += torch.randn(
                identity_reprojection_loss.shape).cuda() * 0.00001
            combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            to_optimise, idxs = torch.min(combined, dim=1)
            
            if mask is not None:
                # ambiguity mask
                to_optimise = to_optimise * mask # DIFFERENCE

            # if 1 then identity not chosen
            outputs["identity_selection/{}".format(scale)] = (
                idxs > identity_reprojection_loss.shape[1] - 1).float()

            losses["loss_map/{}".format(scale)] = to_optimise
            
            loss += to_optimise.mean()

            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
            
            if self.opt.loss_experiment:
                loss_wo = 0
                to_optimise, idxs = torch.min(reprojection_loss, dim=1)

                if mask is not None:
                    # ambiguity mask
                    to_optimise = to_optimise * mask # DIFFERENCE

                loss_wo += to_optimise.mean()
                total_loss_wo_smooth += loss_wo

                mean_disp = disp.mean(2, True).mean(3, True)
                norm_disp = disp / (mean_disp + 1e-7)
                smooth_loss = get_smooth_loss(norm_disp, color)

                loss_wo += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)
                total_loss_wo += loss_wo 

        total_loss /= self.num_scales
        losses["loss"] = total_loss

        if self.opt.loss_experiment:
            total_loss_wo /= self.num_scales
            losses["loss_wo_identity"] = total_loss_wo
            
            total_loss_wo_smooth /= self.num_scales
            losses["loss_wo_identity_&_smooth"] = total_loss_wo_smooth

        return losses

    @staticmethod
    def extract_ambiguity(ipt):
        grad_r = ipt[:, :, :, :-1] - ipt[:, :, :, 1:]
        grad_b = ipt[:, :, :-1, :] - ipt[:, :, 1:, :]

        grad_l = F.pad(grad_r, (1, 0))
        grad_r = F.pad(grad_r, (0, 1))

        grad_t = F.pad(grad_b, (0, 0, 1, 0))
        grad_b = F.pad(grad_b, (0, 0, 0, 1))

        is_u_same_sign = ((grad_l * grad_r) > 0).any(dim=1, keepdim=True)
        is_v_same_sign = ((grad_t * grad_b) > 0).any(dim=1, keepdim=True)
        is_same_sign = torch.logical_or(is_u_same_sign, is_v_same_sign)

        grad_u = (grad_l.abs() + grad_r.abs()).sum(1, keepdim=True) / 2
        grad_v = (grad_t.abs() + grad_b.abs()).sum(1, keepdim=True) / 2
        grad = torch.sqrt(grad_u ** 2 + grad_v ** 2)

        ambiguity = grad * is_same_sign
        return ambiguity
    
    def compute_ambiguity_mask(self, inputs, outputs,
                               reprojection_loss, scale):
        src_scale = 0
        min_reproj, min_idx = torch.min(reprojection_loss, dim=1)

        target_ambiguity = self.extract_ambiguity(inputs[("color", 0, src_scale)])

        reproj_ambiguities = []
        for f_i in self.opt.frame_ids[1:]:
            src_ambiguity = self.extract_ambiguity(inputs[("color", f_i, src_scale)])

            reproj_ambiguity = F.grid_sample(
                src_ambiguity, outputs[("sample", f_i, scale)],
                padding_mode="border", align_corners=True)
            reproj_ambiguities.append(reproj_ambiguity)

        reproj_ambiguities = torch.cat(reproj_ambiguities, dim=1)
        reproj_ambiguity = torch.gather(reproj_ambiguities, 1, min_idx.unsqueeze(1))

        synthetic_ambiguity, _ = torch.cat(
            [target_ambiguity, reproj_ambiguity], dim=1).max(dim=1)

        # if self.opt.ambiguity_by_negative_exponential: # False
        #     ambiguity_mask = torch.exp(-self.opt.negative_exponential_coefficient
        #                                * synthetic_ambiguity)
        # else:
        ambiguity_thresh = 0.3
        ambiguity_mask = synthetic_ambiguity < ambiguity_thresh
        return ambiguity_mask

    def tensor_augmentation(self, input, rotate_angle, crop_factor, do_flip):
        # rotate
        rotate_angle = rotate_angle * math.pi / 180
        rot_mat = torch.tensor([[torch.cos(rotate_angle), -torch.sin(rotate_angle), 0],
                                [torch.sin(rotate_angle), torch.cos(rotate_angle), 0]])
        rot_mat = rot_mat[None, ...].repeat(input.shape[0], 1, 1)
        grid = F.affine_grid(rot_mat, input.size()).to(self.opt.device)
        input = F.grid_sample(input, grid).squeeze(0)

        # crop
        x = int(crop_factor * (input.shape[2] - self.width))
        y = int(crop_factor * (input.shape[1] - self.height))
        input = input[:, y:y+self.height, x:x+self.width]

        # flip
        if do_flip:
            input = torch.flip(input, [2])

        return input.unsqueeze(0)

    def log_loop_metrics(self, metrics, prefix=None):
        writer = self.writer

        def _to_scalar(value):
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return value.detach().cpu().item()
                return None
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.item()
                return None
            if isinstance(value, np.generic):
                return value.item()
            return value

        for metric_name, metric_value in metrics.items():
            scalar_value = _to_scalar(metric_value)
            if scalar_value is None:
                continue
            label = metric_name if prefix is None else "{}/{}".format(prefix, metric_name)
            writer.add_scalar("{}".format(label), scalar_value, self.step)

    def log(self, inputs, outputs, errors, losses, save_images):
        """Write an event to the configured experiment logger
        """
        self.log_loop_metrics(errors)

        for loss_name, loss_value in losses.items():
            if 'map' in loss_name:
                continue
            self.log_loop_metrics({loss_name: loss_value})

        writer = self.writer
            
        # img_entropy = calculate_batch_image_entropy(inputs["color_uncrop", 0, 0]) 
        # img_edge_density = calculate_batch_edge_density(inputs["color_uncrop", 0, 0])
        # writer.add_scalar("img_entropy", img_entropy, self.step)
        # writer.add_scalar("img_edge_density", img_edge_density, self.step)

        gt_height, gt_width = inputs["depth_gt_uncrop"].shape[-2], inputs["depth_gt_uncrop"].shape[-1]
        mask = torch.logical_and(inputs["depth_gt_uncrop"] > self.opt.MIN_DEPTH, inputs["depth_gt_uncrop"] < self.opt.MAX_DEPTH)
        _depth_pred = F.interpolate(outputs["depth"], [gt_height, gt_width], mode="bilinear", align_corners=False)
        _depth_gt = inputs["depth_gt_uncrop"][mask]
        _depth_pred = _depth_pred[mask]
        median_ratio = (torch.median(_depth_gt)/torch.median(_depth_pred)).cpu().item()
        writer.add_scalar("median_ratio", median_ratio, self.step)


        if not save_images:
            return

        j = 0
        s = 0
        frame_id = 0

        cmap = matplotlib.cm.plasma
        cmap.set_bad('white',1.)
        
        cmap_error = matplotlib.cm.viridis
        cmap_error.set_bad('white',1.)
 
        if self.step < 2:
            # Add colorbars for depth and error visualizations
            writer.add_image("vis/depth_colorbar", create_colorbar(vmin=self.opt.MIN_DEPTH, 
                                                                   vmax=self.opt.MAX_DEPTH, 
                                                                   title="Depth (m)"), self.step) 

        # GT depth
        depth_gt = copy.deepcopy(inputs["depth_gt_uncrop"])
        depth_gt = F.interpolate(depth_gt, outputs["depth"].shape[-2:], mode="bilinear", align_corners=False)
        mask = torch.logical_and(depth_gt > self.opt.MIN_DEPTH, depth_gt < self.opt.MAX_DEPTH)
        depth_gt[torch.logical_not(mask)] = np.nan
        depth_gt = depth_gt.squeeze().detach().cpu().numpy()
        depth_gt = np.where(depth_gt, depth_gt, np.nan)
        nan_mask = np.isnan(depth_gt)
        _depth_gt = (255 * cmap(depth_gt/80)).astype('uint8')[:,:,:3]
        # make nan values gray
        _depth_gt[nan_mask, :] = [128, 128, 128]
        _depth_gt = np.transpose(_depth_gt, (2,0,1))
        writer.add_image("vis/depth_gt_{}".format(j),
                         _depth_gt,
                         self.step)
        
        error_max = self.opt.MAX_DEPTH
        error_min = -self.opt.MAX_DEPTH
        epsilon = 1e-6
        log_error_max = np.sign(error_max) * np.log1p(np.abs(error_max) + epsilon)
        log_error_min = np.sign(error_min) * np.log1p(np.abs(error_min) + epsilon)
        pred_keys_for_vis = ['depth', 'depth_model_pred', 'depth_aug_pred']
        for key in pred_keys_for_vis:
            if key not in outputs.keys():
                continue

            # prediction
            pred = copy.deepcopy(outputs[key][j].squeeze().detach().cpu().numpy())
            pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
            pred = np.transpose(pred, (2,0,1))
            writer.add_image("vis/{}_{}".format(key, s),
                            pred,
                            self.step)
            # error
            error_map = outputs[key][j].squeeze().detach().cpu().numpy() - depth_gt
            # Apply logarithmic transformation
            # Add a small epsilon to avoid log(0) and handle negative values
            log_error_map = np.sign(error_map) * np.log1p(np.abs(error_map) + epsilon)
            # Normalize using log-transformed values
            error_vis = (log_error_map - log_error_min) / (log_error_max - log_error_min)
            # Convert to RGB using colormap
            error_vis = (255 * cmap_error(error_vis)).astype('uint8')[:,:,:3]
            error_vis[nan_mask, :] = [128, 128, 128]
            error_vis = np.transpose(error_vis, (2,0,1))
            writer.add_image("vis/{}_error_{}".format(key, s),
                            error_vis,
                            self.step)
                

            # prediction median scaled
            pred = copy.deepcopy(outputs[key][j].squeeze().detach().cpu().numpy()) * median_ratio
            pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
            pred = np.transpose(pred, (2,0,1))
            writer.add_image("vis/gt_median_scaled/{}_{}".format(key, s),
                            pred,
                            self.step)
            
            # errors median scaled
            error_map = outputs[key][j].squeeze().detach().cpu().numpy() * median_ratio - depth_gt
            log_error_map = np.sign(error_map) * np.log1p(np.abs(error_map) + epsilon)
            error_vis = (log_error_map - log_error_min) / (log_error_max - log_error_min)
            # Convert to RGB using colormap
            error_vis = (255 * cmap_error(error_vis)).astype('uint8')[:,:,:3]
            error_vis[nan_mask, :] = [128, 128, 128]
            error_vis = np.transpose(error_vis, (2,0,1))
            writer.add_image("vis/gt_median_scaled/{}_error_{}".format(key, s),
                            error_vis,
                            self.step)
        

        if self.step < 2:
            writer.add_image("vis/depth_error_colorbar", 
                            create_colorbar(vmin=error_min, 
                                            vmax=error_max,
                                            cmap=cmap_error,
                                            title="Depth Error (Log Scale)",
                                            log_scale=True), self.step)

        writer.add_image(
            "vis/color_{}_{}_{}".format(frame_id, s, j),
            inputs[("color_uncrop", frame_id, s)][j].data, self.step)

        if "pseudo_depth_sup" in outputs.keys():
            # pseudo1
            pred = copy.deepcopy(outputs["pseudo_depth_sup"][j]).squeeze().detach().cpu().numpy()
            pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
            pred = np.transpose(pred, (2,0,1))
            writer.add_image("vis/pseudo_sup_{}_{}".format(s, j),
                            pred,
                            self.step)

        if "pseudo_depth_unsup" in outputs.keys():
            # pseudo2
            pred = copy.deepcopy(outputs["pseudo_depth_unsup"][j]).squeeze().detach().cpu().numpy()
            pred = (255 * cmap(pred/80)).astype('uint8')[:,:,:3]
            pred = np.transpose(pred, (2,0,1))
            writer.add_image("vis/pseudo_unsup_{}_{}".format(s, j),
                            pred,
                            self.step)

    def interpolate_gt_depth(self, depth_gt, valid_mask):
        # Create interpolated version of GT depth
        depth_gt_interp = copy.deepcopy(depth_gt)
        
        # Get coordinates of valid and invalid pixels
        y_valid, x_valid = np.where(valid_mask)
        y_invalid, x_invalid = np.where(~valid_mask)
        
        # Only interpolate if we have both valid and invalid pixels
        if len(y_valid) > 0 and len(y_invalid) > 0:
            # Get valid depth values
            valid_values = depth_gt_interp[valid_mask]
            
            # Create interpolator using valid pixels
            
            # Try linear interpolation first
            interp = LinearNDInterpolator(
                np.column_stack([y_valid, x_valid]), 
                valid_values
            )
            
            # Apply interpolation to invalid pixels
            invalid_coords = np.column_stack([y_invalid, x_invalid])
            interp_values = interp(invalid_coords)
            
            # For any remaining NaN values, use nearest neighbor
            nan_mask = np.isnan(interp_values)
            if np.any(nan_mask):
                nn_interp = NearestNDInterpolator(
                    np.column_stack([y_valid, x_valid]), 
                    valid_values
                )
                interp_values[nan_mask] = nn_interp(invalid_coords[nan_mask])
            
            # Fill in the interpolated values
            depth_gt_interp[~valid_mask] = interp_values
        
        return depth_gt_interp

if __name__ == "__main__":
    adapt = Adapt(opts)
    adapt.run_adapt()
