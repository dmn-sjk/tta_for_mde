# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import cv2
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

import pdb


DEPTH_METRIC_NAMES = ["sup/abs_rel", "sup/sq_rel", "sup/rms", "sup/log_rms",
                            "sup/a1", "sup/a2", "sup/a3", "median/global"]
DEPTH_METRIC_NAMES_LOCAL = ["supl/abs_rel", "supl/sq_rel", "supl/rms", "supl/log_rms",
                                    "supl/a1", "supl/a2", "supl/a3", "median/local"]
DEPTH_METRIC_NAMES_UNSUP = ["unsupl/abs_rel", "unsupl/sq_rel", "unsupl/rms",
                                        "unsupl/log_rms", "unsupl/a1", "unsupl/a2", "unsupl/a3",
                                        "median/unsup"]
DEPTH_METRICS = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def depth_to_disp(depth, min_depth, max_depth):
    """Convert groundtruth depth into network's sigmoid output
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = 1 / depth
    disp = (scaled_disp - min_disp) / (max_disp - min_disp)

    return disp


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0], 4, 4).to(device=translation_vector.device)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0], 4, 4)).to(device=vec.device)

    rot[:, 0, 0] = torch.squeeze(x * xC + ca)
    rot[:, 0, 1] = torch.squeeze(xyC - zs)
    rot[:, 0, 2] = torch.squeeze(zxC + ys)
    rot[:, 1, 0] = torch.squeeze(xyC + zs)
    rot[:, 1, 1] = torch.squeeze(y * yC + ca)
    rot[:, 1, 2] = torch.squeeze(yzC - xs)
    rot[:, 2, 0] = torch.squeeze(zxC - ys)
    rot[:, 2, 1] = torch.squeeze(yzC + xs)
    rot[:, 2, 2] = torch.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


def update_ema_variables(model, ema_model, alpha, alpha_prompt=None, global_step=None):
    # Use the true average until the exponential average is more correct
    if global_step:
        alpha = min(1 - 1 / (global_step + 1), alpha)

    for (ema_name, ema_param), (name, param) in zip(ema_model.named_parameters(), model.named_parameters()):
        if alpha_prompt and 'prompt' in name:
            ema_param.data[:] = alpha_prompt * ema_param[:].data[:] + (1 - alpha_prompt) * param[:].data[:]
        else:
            ema_param.data[:] = alpha * ema_param[:].data[:] + (1 - alpha) * param[:].data[:]


class ConvBlock(nn.Module):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    x_ref = torch.tensor(x_ref, dtype=torch.float32).to(depth_ref.device)
    y_ref = torch.tensor(y_ref, dtype=torch.float32).to(depth_ref.device)
    # reference 3D space
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref),
                           torch.stack((x_ref, y_ref, torch.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                           torch.cat((xyz_ref, torch.ones_like(x_ref).unsqueeze(0))))[:3]
    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([1, 1, height, width])
    y_src = xy_src[1].reshape([1, 1, height, width])
    grid = torch.cat((x_src, y_src), 1)
    grid[:, 0, :, :] = 2.0 * grid[:, 0, :, :].clone() / max(width - 1, 1) - 1.0
    grid[:, 1, :, :] = 2.0 * grid[:, 1, :, :].clone() / max(height - 1, 1) - 1.0
    sampled_depth_src = F.grid_sample(depth_src.reshape([1, 1, height, width]), grid.permute(0, 2, 3, 1), 'nearest')
    sampled_depth_src = sampled_depth_src.reshape([height, width])
    # sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                           torch.cat((xy_src, torch.ones_like(x_ref).unsqueeze(0))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)),
                                   torch.cat((xyz_src, torch.ones_like(x_ref).unsqueeze(0))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width])
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width])
    y_reprojected = xy_reprojected[1].reshape([height, width])

    return depth_reprojected, x_reprojected, y_reprojected, xyz_ref, xyz_src


def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref = torch.tensor(x_ref, dtype=torch.float32).to(depth_ref.device)
    y_ref = torch.tensor(y_ref, dtype=torch.float32).to(depth_ref.device)
    depth_reprojected, x2d_reprojected, y2d_reprojected, xyz_ref, xyz_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                     depth_src, intrinsics_src, extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask1 = dist < 1
    mask2 = relative_depth_diff < 0.01
    mask = mask1 * mask2
    # mask = torch.logical_and(dist < 1, relative_depth_diff < 0.01)
    # depth_reprojected[~mask] = 0

    return mask, xyz_ref, xyz_src


class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def get_smooth_loss(disp, img, factor=1):
    """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
    grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
    grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

    grad_disp_x *= torch.exp(-factor * grad_img_x)
    grad_disp_y *= torch.exp(-factor * grad_img_y)

    return grad_disp_x.mean() + grad_disp_y.mean()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class ScaleRecovery(nn.Module):
    """Layer to estimate scale through dense geometrical constrain
    """
    def __init__(self, batch_size, height, width):
        super(ScaleRecovery, self).__init__()
        self.backproject_depth = BackprojectDepth(batch_size, height, width)
        self.batch_size = batch_size
        self.height = height
        self.width = width

    # derived from https://github.com/zhenheny/LEGO
    def get_surface_normal(self, cam_points, nei=1):
        cam_points_ctr  = cam_points[:, :-1, nei:-nei, nei:-nei]
        cam_points_x0   = cam_points[:, :-1, nei:-nei, 0:-(2*nei)]
        cam_points_y0   = cam_points[:, :-1, 0:-(2*nei), nei:-nei]
        cam_points_x1   = cam_points[:, :-1, nei:-nei, 2*nei:]
        cam_points_y1   = cam_points[:, :-1, 2*nei:, nei:-nei]
        cam_points_x0y0 = cam_points[:, :-1, 0:-(2*nei), 0:-(2*nei)]
        cam_points_x0y1 = cam_points[:, :-1, 2*nei:, 0:-(2*nei)]
        cam_points_x1y0 = cam_points[:, :-1, 0:-(2*nei), 2*nei:]
        cam_points_x1y1 = cam_points[:, :-1, 2*nei:, 2*nei:]

        vector_x0   = cam_points_x0   - cam_points_ctr
        vector_y0   = cam_points_y0   - cam_points_ctr
        vector_x1   = cam_points_x1   - cam_points_ctr
        vector_y1   = cam_points_y1   - cam_points_ctr
        vector_x0y0 = cam_points_x0y0 - cam_points_ctr
        vector_x0y1 = cam_points_x0y1 - cam_points_ctr
        vector_x1y0 = cam_points_x1y0 - cam_points_ctr
        vector_x1y1 = cam_points_x1y1 - cam_points_ctr

        normal_0 = F.normalize(torch.cross(vector_x0,   vector_y0,   dim=1), dim=1).unsqueeze(0)
        normal_1 = F.normalize(torch.cross(vector_x1,   vector_y1,   dim=1), dim=1).unsqueeze(0)
        normal_2 = F.normalize(torch.cross(vector_x0y0, vector_x0y1, dim=1), dim=1).unsqueeze(0)
        normal_3 = F.normalize(torch.cross(vector_x1y0, vector_x1y1, dim=1), dim=1).unsqueeze(0)

        normals = torch.cat((normal_0, normal_1, normal_2, normal_3), dim=0).mean(0)
        normals = F.normalize(normals, dim=1)

        refl = nn.ReflectionPad2d(nei)
        normals = refl(normals)

        return normals

    def get_ground_mask(self, cam_points, normal_map, threshold=5, return_cos=False):
        b, _, h, w = normal_map.size()
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        threshold = math.cos(math.radians(threshold))
        ones, zeros = torch.ones(b, 1, h, w).cuda(), torch.zeros(b, 1, h, w).cuda()
        vertical = torch.cat((zeros, ones, zeros), dim=1)

        cosine_sim = cos(normal_map, vertical).unsqueeze(1)
        vertical_mask = (cosine_sim > threshold) | (cosine_sim < -threshold)

        y = cam_points[:,1,:,:].unsqueeze(1)
        ground_mask = vertical_mask.masked_fill(y <= 0, False)

        if return_cos:
            return ground_mask, cosine_sim
        else:
            return ground_mask

    def forward(self, depth, K, real_cam_height, ground_mask=None):
        inv_K = torch.inverse(K)

        cam_points = self.backproject_depth(depth, inv_K).reshape(self.batch_size, 4, self.height, self.width)
        surface_normal = self.get_surface_normal(cam_points)
        if ground_mask is None:
            ground_mask = self.get_ground_mask(cam_points, surface_normal)
        else:
            ground_mask_calc, cosine_sim = self.get_ground_mask(cam_points,
                                                                surface_normal,
                                                                return_cos=True)
            ground_mask *= ground_mask_calc

        cam_heights = (cam_points[:,:-1,:,:] * surface_normal).sum(1).abs().unsqueeze(1)
        cam_heights_masked = torch.masked_select(cam_heights, ground_mask)
        # cam_height = torch.median(cam_heights_masked).unsqueeze(0)
        cam_height = torch.mean(cam_heights_masked).unsqueeze(0)

        scale = torch.reciprocal(cam_height).mul_(real_cam_height)

        if np.isnan(scale.detach().cpu().numpy()):
            scale[0] = 1

        if ground_mask is None:
            return scale, cam_heights*ground_mask
        else:
            return scale, cam_heights*ground_mask, cosine_sim


def compute_depth_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean((gt - pred) ** 2 / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def preprocess_depth_values(opt, depth_gt, depth_pred, median_scaling=False):
    gt_height, gt_width = depth_gt.shape[2], depth_gt.shape[3]
    mask = torch.logical_and(depth_gt > opt.MIN_DEPTH, depth_gt < opt.MAX_DEPTH)
    _depth_pred = F.interpolate(depth_pred, [gt_height, gt_width], mode="bilinear", align_corners=False)

    if opt.dataset == "kitti":
        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, int(0.40810811*gt_height):int(0.99189189*gt_height),
                    int(0.03594771*gt_width):int(0.96405229*gt_width)] = 1
        mask = mask * crop_mask

    # mask0 = torch.ones_like(mask)
    # mask0[:, :, :int(0.3*depth_gt.shape[2]), :] = 0
    # mask = mask * mask0

    _depth_gt = depth_gt[mask]
    _depth_pred = _depth_pred[mask]
    median_ratio = torch.median(_depth_gt)/torch.median(_depth_pred)
    if median_scaling:
        _depth_pred *= median_ratio

    _depth_gt = torch.clamp(_depth_gt, min=opt.MIN_DEPTH, max=opt.MAX_DEPTH)
    _depth_pred = torch.clamp(_depth_pred, min=opt.MIN_DEPTH, max=opt.MAX_DEPTH)

    return _depth_gt, _depth_pred, median_ratio

def compute_depth_errors_adadepth(opt, depth_gt, depth_pred, median_scaling=False):
    _depth_gt, _depth_pred, median_ratio = preprocess_depth_values(opt, 
                                                                   depth_gt, 
                                                                   depth_pred, 
                                                                   median_scaling=median_scaling)
    
    abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(_depth_gt, _depth_pred)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, median_ratio


class AvgDepthErrorsPerDepthRange:
    def __init__(self, opt, num_bins=8, median_scaling=False):
        self.opt = opt
        self.num_bins = num_bins
        self.median_scaling = median_scaling
    
        # Create depth bins with less aggressive logarithmic spacing
        min_depth = opt.MIN_DEPTH
        max_depth = opt.MAX_DEPTH

        # Use a hybrid approach: combine linear and logarithmic spacing
        # This creates less aggressive spacing than pure logarithmic
        alpha = 0.55  # Controls the balance: 0 = linear, 1 = logarithmic
        
        # Linear spacing
        linear_bins = np.linspace(min_depth, max_depth, num_bins + 1)
        
        # Logarithmic spacing
        log_min = np.log(min_depth)
        log_max = np.log(max_depth)
        log_bins = np.linspace(log_min, log_max, num_bins + 1)
        log_bins = np.exp(log_bins)
        
        # Combine linear and logarithmic with alpha weight
        self.depth_bins = alpha * log_bins + (1 - alpha) * linear_bins
        
        self.bins_error_sums = [{metric: 0 for metric in DEPTH_METRICS} for _ in range(num_bins)]
        self.bins_image_counts = [0 for _ in range(num_bins)]
        self.bins_pixel_counts = [0 for _ in range(num_bins)]  # Track pixel counts per bin
    
    def get_depth_bins(self) -> List[float]:
        return self.depth_bins
        
    def register_data(self, depth_gt, depth_pred):
        gt, pred, _ = preprocess_depth_values(self.opt, 
                                              depth_gt, 
                                              depth_pred, 
                                              median_scaling=self.median_scaling)

        # Process each depth bin
        for i in range(self.num_bins):
            # Create mask for current depth bin
            if i == 0:
                # First bin: min_depth to depth_bins[1]
                bin_mask = torch.logical_and(gt >= self.depth_bins[i], gt < self.depth_bins[i+1])
            elif i == self.num_bins - 1:
                # Last bin: depth_bins[7] to max_depth
                bin_mask = torch.logical_and(gt >= self.depth_bins[i], gt <= self.depth_bins[i+1])
            else:
                # Middle bins: depth_bins[i] to depth_bins[i+1]
                bin_mask = torch.logical_and(gt >= self.depth_bins[i], gt < self.depth_bins[i+1])
            
            # Get valid pixels in this bin
            gt_bin = gt[bin_mask]
            pred_bin = pred[bin_mask]
            
            # Count pixels in this bin
            num_pixels = gt_bin.shape[0]
            self.bins_pixel_counts[i] += num_pixels  # Track total pixel count
            
            # Skip if no pixels in this bin
            if num_pixels == 0:
                continue
            
            # Calculate errors for this bin
            errors = list(compute_depth_errors(gt_bin, pred_bin))
            for idx, term in enumerate(errors):
                errors[idx] = term.detach().cpu().numpy()
            
            for metric_idx, metric in enumerate(DEPTH_METRICS):
                self.bins_error_sums[i][metric] += errors[metric_idx]
            self.bins_image_counts[i] += 1

    def get_avg_errors(self) -> List[Dict[str, float]]:
        avg_errors_per_bin = []
        for i in range(self.num_bins):
            avg_errors_per_bin.append({metric: (self.bins_error_sums[i][metric] / self.bins_image_counts[i]) \
                if self.bins_image_counts[i] > 0 else 0 for metric in DEPTH_METRICS})
        return avg_errors_per_bin

    def log_charts(self, writer, step):
        """Log charts showing errors per depth bin to the configured backend"""
        if all(count == 0 for count in self.bins_image_counts):
            return
        
        import matplotlib.pyplot as plt
        
        avg_errors_per_bin = self.get_avg_errors()
        
        # Create depth bin labels (midpoints of each bin)
        bin_labels = []
        for i in range(self.num_bins):
            if i == 0:
                bin_labels.append(f"{self.depth_bins[i]:.1f}-{self.depth_bins[i+1]:.1f}")
            elif i == self.num_bins - 1:
                bin_labels.append(f"{self.depth_bins[i]:.1f}-{self.depth_bins[i+1]:.1f}")
            else:
                bin_labels.append(f"{self.depth_bins[i]:.1f}-{self.depth_bins[i+1]:.1f}")
        
        # Create charts for each metric
        for metric in DEPTH_METRICS:
            # Extract values for this metric
            values = [bin_errors[metric] for bin_errors in avg_errors_per_bin]
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            bars = ax.bar(range(len(bin_labels)), values, alpha=0.7, color='skyblue', edgecolor='navy')
            
            # Customize the chart
            ax.set_xlabel('Depth Range (m)', fontsize=12)
            ax.set_ylabel(metric.upper(), fontsize=12)
            name = f'{metric.upper()} Error by Depth Range' if not self.median_scaling else \
                f'{metric.upper()} Error by Depth Range (Median Scaling)'
            ax.set_title(name, fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(bin_labels)))
            ax.set_xticklabels(bin_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            
            if not self.median_scaling:
                writer.add_figure(f'depth_range_errors/{metric}_by_bin', fig, step)
            else:
                writer.add_figure(f'depth_range_errors_median_scaling/{metric}_by_bin', fig, step)
            plt.close()
        
        # Create pixel count chart
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(bin_labels)), self.bins_pixel_counts, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        
        # Customize the chart
        ax.set_xlabel('Depth Range (m)', fontsize=12)
        ax.set_ylabel('Pixel Count', fontsize=12)
        name = 'Pixel Count by Depth Range'
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, self.bins_pixel_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        writer.add_figure(f'depth_range_errors/pixel_count_by_bin', fig, step)
        plt.close()
 
class SingleStageReconHead(nn.Module):
    def __init__(self, in_channels=1024, hidden_channels=128):
        """
        Reconstruction head using only Stage 4 features.
        Args:
            in_channels (int): Number of channels in Stage 4 features (e.g., 1024 for Swin-B).
            hidden_channels (int): Number of channels after channel reduction.
        """
        super(SingleStageReconHead, self).__init__()
        # Channel reduction
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, stride=1)
        # Upsampling layers
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        # Output layer (RGB: 3 channels)
        self.conv_out = nn.Conv2d(hidden_channels, 3, kernel_size=3, stride=1, padding=1)
        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, stage4_features):
        """
        Forward pass to reconstruct coarse RGB image from Stage 4 features.
        Args:
            stage4_features (torch.Tensor): Stage 4 features (B, C4, H/32, W/32).
        Returns:
            torch.Tensor: Reconstructed RGB image (B, 3, H/8, W/8).
        """
        # Channel reduction
        x = self.conv1(stage4_features)  # (B, hidden_channels, H/32, W/32)
        x = F.relu(x)
        # Upsample to H/16 x W/16
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, hidden_channels, H/16, W/16)
        x = self.conv2(x)
        x = F.relu(x)
        # Upsample to H/8 x W/8
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, hidden_channels, H/8, W/8)
        x = self.conv3(x)
        x = F.relu(x)
        # Output RGB
        x = self.conv_out(x)  # (B, 3, H/8, W/8)
        x = self.sigmoid(x)  # Normalize to [0, 1]
        return x

class MultiStageReconHead(nn.Module):
    def __init__(self, stage4_channels=1024, stage3_channels=384, hidden_channels=128):
        """
        Reconstruction head using Stage 3 and Stage 4 features.
        Args:
            stage4_channels (int): Number of channels in Stage 4 features (e.g., 1024 for Swin-B).
            stage3_channels (int): Number of channels in Stage 3 features (e.g., 384 for Swin-B).
            hidden_channels (int): Number of channels after channel reduction.
        """
        super(MultiStageReconHead, self).__init__()
        # Channel reduction for Stage 4
        self.conv_stage4 = nn.Conv2d(stage4_channels, hidden_channels, kernel_size=1, stride=1)
        # Channel reduction for Stage 3
        self.conv_stage3 = nn.Conv2d(stage3_channels, hidden_channels, kernel_size=1, stride=1)
        # Fusion and refinement
        self.conv_fuse = nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, stride=1, padding=1)
        self.conv_refine = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)
        # Output layer (RGB: 3 channels)
        self.conv_out = nn.Conv2d(hidden_channels, 3, kernel_size=3, stride=1, padding=1)
        # Activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, stage3_features, stage4_features):
        """
        Forward pass to reconstruct coarse RGB image from Stage 3 and Stage 4 features.
        Args:
            stage3_features (torch.Tensor): Stage 3 features (B, C3, H/16, W/16).
            stage4_features (torch.Tensor): Stage 4 features (B, C4, H/32, W/32).
        Returns:
            torch.Tensor: Reconstructed RGB image (B, 3, H/8, W/8).
        """
        # Process Stage 4: Upsample and reduce channels
        x4 = self.conv_stage4(stage4_features)  # (B, hidden_channels, H/32, W/32)
        x4 = F.relu(x4)
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)  # (B, hidden_channels, H/16, W/16)
        
        # Process Stage 3: Reduce channels
        x3 = self.conv_stage3(stage3_features)  # (B, hidden_channels, H/16, W/16)
        x3 = F.relu(x3)
        
        # Fuse Stage 3 and Stage 4
        x = torch.cat([x3, x4], dim=1)  # (B, hidden_channels*2, H/16, W/16)
        x = self.conv_fuse(x)  # (B, hidden_channels, H/16, W/16)
        x = F.relu(x)
        
        # Upsample to H/8 x W/8 and refine
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)  # (B, hidden_channels, H/8, W/8)
        x = self.conv_refine(x)
        x = F.relu(x)
        
        # Output RGB
        x = self.conv_out(x)  # (B, 3, H/8, W/8)
        x = self.sigmoid(x)  # Normalize to [0, 1]
        return x
