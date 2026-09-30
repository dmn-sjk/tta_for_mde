import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import PIL.Image as pil
from skimage import transform
import random
import copy
import sys

# add shift module to path
# shift_root_dir = os.path.abspath(os.path.dirname(__file__))
# sys.path.append(shift_root_dir)

from shift_dev import SHIFTDataset as _SHIFTDataset
from shift_dev.types import Keys
from shift_dev.utils.backend import ZipBackend
from utils.utils import save_tensor_as_image


class SHIFTDataset:
    """
    Shift dataset class

    Parameters
    ----------
    path : str
        Path to the dataset
    split : str {'train', 'val', 'test'}
        Which dataset split to use
    cameras : list of str
        Which cameras to get information from
    depth_type : str
        Which lidar will be used to generate ground-truth information
    back_context : int
        Size of the backward context
    forward_context : int
        Size of the forward context
    data_transform : Function
        Transformations applied to the sample
    """
    
    # use KITTI height
    CAM_H = 1.65

    def __init__(self, path, _,
                 height=None,
                 width=None,
                 frame_idxs=[0, -1, 1],
                 num_scales=4,
                 is_train=False,
                 flip_aug=True,
                 rotate_aug=True,
                 pseu=False,
                 tgt_height=1.65,
                 tgt_focal=720,
                 depth_path=None,
                 opt=None,
                 gt_transformation=False,
                 **kwargs
                 ):
        self.opt = opt
        self.path = path
        # default target KITTI dataset
        self.tgt_height = tgt_height
        self.tgt_focal = tgt_focal
        self.pseu = pseu
        self.gt_transformation = gt_transformation

        self.frame_idxs = frame_idxs

        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.is_train = is_train
        self.flip_aug = flip_aug
        self.rotate_aug = rotate_aug
        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1
        self.data_transform = transforms.ColorJitter.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)
        self.to_tensor = transforms.ToTensor()
        self.normalise = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if self.opt.dataset == 'shift_discr1:11' or self.opt.dataset == 'shift':
            framerate = "videos"
            split = "val"
            shift_type = "discrete"
        else:
            raise NotImplementedError(f"{self.opt.dataset} not implemented")
            
            
        self.dataset = _SHIFTDataset(
            data_root=path,
            split=split,
            framerate=framerate,
            keys_to_load=[
                Keys.images,                # note: images, shape (1, 3, H, W), uint8 (RGB)
                Keys.boxes2d,
                Keys.intrinsics,            # note: camera intrinsics, shape (3, 3)
                Keys.depth_maps,            # note: depth maps, shape (1, H, W), float (meters)
            ],
            views_to_load=["front"],
            shift_type=shift_type,          # also supports "continuous/1x", "continuous/10x", "continuous/100x"
            backend=ZipBackend(),           # also supports HDF5Backend(), FileBackend()
            verbose=True,
        )

        # Print the dataset size
        print(f"Total number of samples: {len(self.dataset)}.")
        
        # modify the indices to account for the backward and forward context
        video_to_indices = self.dataset.video_to_indices
        self.frame_indices = []
        back_context = abs(min(frame_idxs)) if min(frame_idxs) < 0 else 0
        forward_context = max(frame_idxs) if max(frame_idxs) > 0 else 0
 
        if self.opt.dataset == 'shift_discr1:11':
            # first 10 videos omitting the first one (static video)
            chosen_seqs = list(video_to_indices.items())[1:11]
        else:
            chosen_seqs = list(video_to_indices.items())
        
        for _, indices in chosen_seqs:
            _indices = indices[back_context:len(indices)-forward_context]
            self.frame_indices.extend(_indices)
        
        # limit the number of samples to 10000
        # self.frame_indices = self.frame_indices[:10000]

        # # Print the tensor shape of the first batch.
        # dataloader = torch.utils.data.DataLoader(
        #     self.dataset,
        #     batch_size=1,
        #     shuffle=False,
        # )
        # print('\n')
        # for i, batch in enumerate(dataloader):
        #     print(f"Batch {i}:\n")
        #     print(f"{'Item':20} {'Shape':35} {'Min':10} {'Max':10}")
        #     print("-" * 80)
        #     for k, data in batch["front"].items():
        #         if k == 'intrinsics':
        #             print(data)
        #         if isinstance(data, torch.Tensor):
        #             print(f"{k:20} {str(data.shape):35} {data.min():10.2f} {data.max():10.2f}")
        #         else:
        #             print(f"{k:20} {data}")
        #     break

    def get_frame_data(self, idx, frame_id=0, view="front"):
        """Return current timestep of a key from a sensor"""
        data = self.dataset[self.frame_indices[idx] + frame_id][view]
        return data

    def get_colour(self, color, do_flip, rotate_angle, crop_factor, width, height):
        # orig_height, orig_width = color.size[1], color.size[0]

        # color = color.resize(self.full_res, pil.ANTIALIAS)
        # kb crop instead of resizing
        top_margin = int(self.orig_height - self.full_res[1])
        left_margin = int((self.orig_width - self.full_res[0]) / 2)
        color = color.crop((left_margin, top_margin,
                            left_margin + self.full_res[0],
                            top_margin + self.full_res[1]))
        # resize image
        color = color.resize((self.width, self.height), pil.ANTIALIAS)

        # random rotate
        if rotate_angle:
            color = color.rotate(rotate_angle, resample=pil.BILINEAR)

        # random crop
        x = int(crop_factor * (color.size[0] - width))
        y = int(crop_factor * (color.size[1] - height))
        box = (x, y, x + width, y + height)
        color = color.crop(box)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_depth(self, depth_gt, do_flip, rotate_angle, crop_factor, width, height):
        # kb crop
        top_margin = int(self.orig_height - self.full_res[1])
        left_margin = int((self.orig_width - self.full_res[0]) / 2)
        depth_gt = depth_gt[top_margin:top_margin+self.full_res[1],
                            left_margin:left_margin+self.full_res[0]]

        # random rotate
        depth_gt = pil.fromarray(depth_gt*256).convert('I')
        if rotate_angle:
            depth_gt = depth_gt.rotate(rotate_angle, resample=pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        # random crop
        assert depth_gt.shape[0] >= height
        assert depth_gt.shape[1] >= width
        x = int(crop_factor * (depth_gt.shape[1] - width))
        y = int(crop_factor * (depth_gt.shape[0] - height))
        depth_gt = depth_gt[y:y + height,
                            x:x + width]
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])
            if "color_uncrop" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

        for k in list(inputs):
            f = inputs[k]
            if "color_uncrop" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        """Length of dataset"""
        return len(self.frame_indices)
    
    def __getitem__(self, idx):
        """Get a dataset sample"""
        frames = {}
        for i in self.frame_idxs:
            frames[i] = self.get_frame_data(idx, frame_id=i)

        inputs = {}
        inputs['scene'] = frames[0]['videoName']

        # augmentations
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and self.flip_aug and random.random() > 0.5
        do_rotate = self.is_train and self.rotate_aug
        rotate_angle = (random.random() - 0.5) * 2 * 1.0 if do_rotate else 0
        crop_factor = random.random()

        inputs["do_flip"] = do_flip
        inputs["rotate_angle"] = torch.tensor(rotate_angle).type(torch.float32)
        inputs["crop_factor"] = torch.tensor(crop_factor).type(torch.float32)

        # camera intrinsics
        intrinsics_raw = frames[0]['intrinsics']
        inputs["focal_length"] = torch.tensor(intrinsics_raw[0, 0])

        # calculate sizes according to camera params
        if self.opt.scale_alignment:
            self.resize_factor = inputs['focal_length'] * self.CAM_H / (self.tgt_focal * self.tgt_height)
        else:
            self.resize_factor = 1

        gt_size = frames[0]['original_hw']
        self.orig_height, self.orig_width = gt_size[0], gt_size[1]
        # calculate secondary sizes
        # NOTE: divide by 32 (patch size in transformer) and multuply by 32 to make it divisible
        self.width_sec = int(self.orig_width / self.resize_factor) // 32 * 32
        self.height_sec = int(self.orig_height / self.resize_factor) // 32 * 32
        inputs["width_sec"], inputs["height_sec"] = self.width_sec, self.height_sec
        # NOTE: multiplying by resize_factor below is the reverse of applying resize_factor, but after making the shape divisible by 32
        # it is not equal to the original width and height
        self.full_res = (int(self.width_sec * self.resize_factor),
                         int(self.height_sec * self.resize_factor))
        if self.height is None:
            self.height, self.width = self.height_sec, self.width_sec
            self.opt.height, self.opt.width = self.height_sec, self.width_sec
        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        cam_K = np.zeros((4, 4), dtype=np.float32)
        cam_K[:3, :3] = intrinsics_raw
        cam_K[2, 2] = 1
        cam_K[3, 3] = 1
        cam_K[0, :] = cam_K[0, :] / gt_size[1]
        cam_K[1, :] = cam_K[1, :] / gt_size[0]
        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = cam_K.copy()
            K[0, :] *= self.width_sec // (2 ** scale)
            K[1, :] *= self.height_sec // (2 ** scale)
            inv_K = np.linalg.pinv(K)
            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        # rgb images
        for i in self.frame_idxs:
            colour_cur = frames[i]['images'][0]
            # values are in range [0, 255] but in float32, which breakes the conversion to PIL image
            colour_cur = colour_cur.to(torch.uint8)
            colour_cur = transforms.functional.to_pil_image(colour_cur, mode='RGB')
            inputs[("color", i, -1)] = self.get_colour(colour_cur, do_flip,
                                                       rotate_angle, crop_factor,
                                                       self.width, self.height)
            if self.pseu:
                inputs[("color_uncrop", i, -1)] = self.get_colour(colour_cur, False,
                                                                 0, 1,
                                                                 self.width_sec, self.height_sec)
            else:
                inputs[("color_uncrop", i, -1)] = self.get_colour(colour_cur, do_flip,
                                                                 0, 1,
                                                                 self.width_sec, self.height_sec)

        # colour augmentation for the inputs
        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)
        self.preprocess(inputs, color_aug)
        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]
            del inputs[("color_uncrop", i, -1)]
            del inputs[("color_uncrop_aug", i, -1)]
        
        # depth
        depth_raw = frames[0]['depth_maps'][0].numpy()
        # NOTE: don't use this, since the else option does not make sense 
        # (it crops instead of resizing to the image size), explained in DGP dataset
        # if self.height is None:
        #     depth_gt = self.get_depth(depth_raw, do_flip,
        #                               rotate_angle, crop_factor,
        #                               int(self.width * self.resize_factor),
        #                               int(self.height * self.resize_factor))
        # else:
        #     depth_gt = self.get_depth(depth_raw, do_flip,
        #                               rotate_angle, crop_factor,
        #                               self.width, self.height)

        # inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
        # inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        depth_gt_uncrop = self.get_depth(depth_raw, False, 0, 1,
                                         int(self.width_sec * self.resize_factor),
                                         int(self.height_sec * self.resize_factor))
        inputs["depth_gt_uncrop"] = np.expand_dims(depth_gt_uncrop, 0)
        inputs["depth_gt_uncrop"] = torch.from_numpy(inputs["depth_gt_uncrop"].astype(np.float32))
        
        if self.gt_transformation:
            for f_i in self.frame_idxs[1:]:
                transform = self.get_temporal_camera_transform(frames[0], frames[f_i])
                inputs[("gt_cam_T_cam", 0, f_i)] = transform
        
        return inputs

    def get_temporal_camera_transform(self, frame1, frame2):
        T_1_g = frame1['extrinsics']
        T_2_g = frame2['extrinsics']
        T_1_2 = torch.linalg.inv(T_1_g) @ T_2_g

        # fix_handedness = torch.tensor([
        # [-1, 0, 0, 0],  # x_new = -x_old (left -> right)
        # [0, 1, 0, 0],  
        # [0, 0, 1, 0],
        # [0, 0, 0, 1]
        # ], dtype=torch.float64, device=T_1_2.device)
        # T_1_2 = fix_handedness @ T_1_2 @ torch.linalg.inv(fix_handedness)

        # coord_transform = torch.tensor([
        #     [0, 0, -1, 0],  # z_new = -x_old
        #     [0, 1, 0, 0],
        #     [-1, 0, 0, 0],   # x_new = -z_old
        #     [0, 0, 0, 1]
        # ], dtype=torch.float64, device=T_1_2.device)
        # T_1_2 = coord_transform @ T_1_2 @ torch.linalg.inv(coord_transform)
        
        # coord_transform1 = torch.tensor([
        #     [0, 1, 0, 0],  # x_new = y_old
        #     [1, 0, 0, 0],   # y_new = x_old
        #     [0, 0, 1, 0],
        #     [0, 0, 0, 1]
        # ], dtype=torch.float64, device=T_1_2.device)
        # T_1_2 = coord_transform1 @ T_1_2 @ torch.linalg.inv(coord_transform1)


        # translate the axes directions to the standard CV notation
        coord_transform = torch.tensor([
            [0, 1, 0, 0],
            [0, 0, -1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float64, device=T_1_2.device)
        T_1_2 = coord_transform @ T_1_2 @ torch.linalg.inv(coord_transform)
        # this is correct TODO: check why the inverse is correct
        T_1_2 = torch.linalg.inv(T_1_2)

        return T_1_2.to(torch.float32)
        