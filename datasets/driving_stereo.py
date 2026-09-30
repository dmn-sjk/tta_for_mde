import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import torchvision.transforms.functional as F
import torch

from utils.utils import save_tensor_as_image
from .mono_dataset_driving_stereo import MonoDataset



class DrivingStereo(MonoDataset):
    """Superclass for different types of DrivingStereo dataset loader
    """
    
    CAM_H = 1.65 # kitti value, it is not available for this dataset
    
    full_res_shape = (1758, 800)
    
    def __init__(self, *args, **kwargs):
        if self.opt.scale_alignment:
            raise NotImplementedError("Scale alignment not implemented for driving stereo dataset")
        
        super(DrivingStereo, self).__init__(*args, **kwargs)
        
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        
        folder = self.filenames[0].split('/')[0]
        depth_path = f'./splits/driving_stereo/{folder}/gt_depths.npz'
        self.gt_depths = np.load(depth_path)["data"]


    def get_color(self, line, stereo_split=False):
        color = self.loader(os.path.join(self.data_path, line))
        return color
    
    def get_depth(self, line, index, side, do_flip,
                  rotate_angle, crop_factor, width, height):
        
        # TODO: make sure that the GT depth is in the frame of left camera, now asssuming that it is

        depth_gt = self.gt_depths[index]
        depth_gt = pil.fromarray(depth_gt)

        orig_height, orig_width = depth_gt.size[1], depth_gt.size[0]
        # depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        top_margin = int(orig_height - self.full_res_shape[1])
        left_margin = int((orig_width - self.full_res_shape[0]) / 2)
        depth_gt = depth_gt.crop((left_margin, top_margin,
                            left_margin + self.full_res_shape[0],
                            top_margin + self.full_res_shape[1]))

        # random rotate
        if rotate_angle:
            depth_gt = depth_gt.rotate(rotate_angle, resample=pil.NEAREST)

        # according to the website for full resolution depth maps, the depth values should be divieded by 128,
        # but they were already divided by 256 in export_gt_depth.py script, hence we need to multiply by 2
        # TODO: make sure its right
        depth_gt = np.array(depth_gt).astype(np.float32) * 2

        # random crop
        assert depth_gt.shape[0] >= height
        assert depth_gt.shape[1] >= width
        x = int(crop_factor * (depth_gt.shape[1] - width))
        y = int(crop_factor * (depth_gt.shape[0] - height))
        depth_gt = depth_gt[y:y + height, x:x + width]

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def load_intrinsics(self, line):
        calib_filename = line.split('/')[-1].split('_')[0] + '.txt'
        calib_file = os.path.join(self.data_path, 'calib', 'full-image-calib', calib_filename)
        
        file_dic = {}
        with open(calib_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                key = line.split(': ')[0]
                val = line.split(': ')[1]
                file_dic[key] = [float(v) for v in val.split()]

        return file_dic
    
    # def check_depth(self):
    #     # TODO: check if depth is available in splits/domain_name folder
    #     return True
    
    # def get_cam_params(self, side, do_flip, crop_factor, width, height, ):
    #     # TODO: side map since to be different from kitti (there is only 101 and 103, kitti has 00, 01, 02, 03)
    #     # load the camera calibration file
    #     date = self.filenames[0].split()[0].split('/')[0]
    #     cam_params_file = os.path.join(
    #         self.data_path,
    #         date,
    #         "calib_cam_to_cam.txt"
    #     )
    #     file_dic = {}
    #     with open(cam_params_file, 'r') as f:
    #         for line in f.readlines():
    #             line = line.strip('\n')
    #             key = line.split(': ')[0]
    #             val = line.split(': ')[1]
    #             file_dic[key] = val.split()
    #     gt_imgsize = [float(i) for i in file_dic["S_rect_0{}".format(self.side_map[side])]]
    #     self.gt_imgsize = np.array(gt_imgsize)
    #     cam_K = [float(i) for i in file_dic["P_rect_0{}".format(self.side_map[side])]]
    #     self.T_L = np.array([float(i) for i in file_dic["T_0{}".format(self.side_map['l'])]])
    #     self.T_R = np.array([float(i) for i in file_dic["T_0{}".format(self.side_map['r'])]])
    #     cam_K = np.array(cam_K).reshape((3,4))
    #     self.focal_length = cam_K[0, 0]
    #     # move the principal points according to crop factors
    #     # kb crop
    #     top_margin = int(self.gt_imgsize[1] - self.full_res_shape[1])
    #     left_margin = int((self.gt_imgsize[0] - self.full_res_shape[0]) / 2)
    #     cam_K[0, 2] = cam_K[0, 2] - left_margin
    #     cam_K[1, 2] = cam_K[1, 2] - top_margin

    #     # # random crop
    #     # x = int(crop_factor * (self.full_res_shape[0] - width))
    #     # y = int(crop_factor * (self.full_res_shape[1] - height))
    #     # cam_K[0, 2] = cam_K[0, 2] - x
    #     # cam_K[1, 2] = cam_K[1, 2] - y
    #     # if do_flip:
    #     #     cam_K[0, 2] = width - cam_K[0, 2]

    #     cam_K[0, :] = cam_K[0, :] / self.gt_imgsize[0]
    #     cam_K[1, :] = cam_K[1, :] / self.gt_imgsize[1]
    #     cam_K[:, 3] = 0
    #     self.K = np.zeros((4, 4), dtype=np.float32)
    #     self.K[:3, :] = cam_K
    #     self.K[3, 3] = 1
