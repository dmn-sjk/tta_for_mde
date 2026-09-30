import os
import random
from collections import deque

from methods.base import BaseAdaptation
from methods import register_method
from utils.diagnostics import (
    DepthOutBufferDiagnostics,
    compute_depth_out_spectral_diagnostics,
    FeauturesBufferDiagnostics,
    compute_gradient_magnitude_direction_error,
    compute_image_gradient_chamfer_distance,
    compute_residual_total_variation,
    compute_total_variation,
    compute_tv_error,
)
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
import torch.nn as nn
from layers import disp_to_depth, compute_depth_errors_adadepth, compute_depth_errors
from networks import get_supervised_models, get_self_supervised_models
from utils.svdp_augs import SVDPMultiScaleFlipAug, Resize, RandomFlip
from utils.losses import DepthLoss
from layers import update_ema_variables
from augmentations import get_augs

import torch.nn.functional as F
import copy
import lpips
from utils.effective_rank import EffectiveRank, EffectiveRankHoldout
from layers import SSIM
from layers import depth_to_disp


@register_method(name='consistency')
class Consistency(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        if self.opt.model_type == "supervised":
            self.models = get_supervised_models(self.opt)
        else:
            # self.models = get_self_supervised_models(self.opt)
            # TODO: loss calculation is different for self-supervised models
            raise NotImplementedError("Self-supervised models not implemented yet")

        # norm_layers = (
        #     nn.BatchNorm1d,
        #     nn.BatchNorm2d,
        #     nn.BatchNorm3d,
        #     nn.SyncBatchNorm,
        #     nn.InstanceNorm1d,
        #     nn.InstanceNorm2d,
        #     nn.InstanceNorm3d,
        #     nn.GroupNorm,
        #     nn.LayerNorm,
        # )
        for m in self.models.values():
            m.eval()
            # for param in m.parameters():
            #     param.requires_grad = False
            # for module in m.modules():
            #     if isinstance(module, norm_layers):
            #         for param in module.parameters():
            #             param.requires_grad = True
        self.initial_model_states = {
            name: copy.deepcopy(model.state_dict())
            for name, model in self.models.items()
        }
        
        # create ema models
        self.models_ema = copy.deepcopy(self.models)
        for model in self.models_ema.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                param.detach_()
        
        # self.models_source = copy.deepcopy(self.models)
        # for model in self.models_source.values():
        #     model.eval()
        #     for param in model.parameters():
        #         param.requires_grad = False
        #         param.detach_()
        
        # following DenseSiam        
        # hid_channels = 128
        # self.predictor = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, hid_channels, 1, bias=False),
        #     torch.nn.BatchNorm2d(hid_channels), torch.nn.ReLU(inplace=True),
        #     torch.nn.Conv2d(hid_channels, 1, 1))
        # self.predictor = self.predictor.cuda()

        parameters_to_train = []
        parameters_to_train += [
            param for param in self.models["encoder"].parameters()
            if param.requires_grad
        ]
        parameters_to_train += [
            param for param in self.models["depth"].parameters()
            if param.requires_grad
        ]
        # parameters_to_train += list(self.predictor.parameters())
        self.optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)
        self.initial_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
 
        # self.svdp_augs = SVDPMultiScaleFlipAug(
        #     img_scale=(self.opt.width, self.opt.height), 
        #     img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        #     flip=False,
        #     patch_size=self.opt.patch_size,
        #     transforms=[
        #         Resize(keep_ratio=True),
        #         RandomFlip(),
        #         # dict(type='Normalize', **img_norm_cfg), # assume the images are already normalized form dataloader
        #     ])
        
        self.augs = get_augs(self.opt)
        self.loss_func = DepthLoss(loss='l1')
        self.depth_out_buffer_diagnostics = DepthOutBufferDiagnostics(
            maxlen=getattr(self.opt, "depth_out_analysis_buffer_size", 50)
        )
        self.feature_buffer_diagnostics = FeauturesBufferDiagnostics(
            maxlen=getattr(self.opt, "feat_analysis_buffer_size", 50)
        )
        self.effective_rank_holdout_path = os.path.join(
            "playground_new", "effective_rank_holdout",
            f"effective_rank_holdout_{self.opt.dataset}.pt",
        )
        # self.effective_rank_holdout = EffectiveRankHoldout(
        #     holdout_path=self.effective_rank_holdout_path,
        #     mean=self.opt.mean,
        #     std=self.opt.std,
        #     holdout_every_n=getattr(self.opt, "effective_rank_holdout_every_n", 23),
        #     max_images=getattr(self.opt, "effective_rank_holdout_max_images", 50),
        # )

        self.er = EffectiveRank(holdout_path=self.effective_rank_holdout_path, device=self.opt.device)
        self.step = 0

        self.ssim = SSIM()
        self.ssim.to(self.opt.device)
        
        self.steps_since_reset = 0
        self.reset_every = int(self.opt.alpha)
        
    def get_aug_image(self, img):
        aug_x = self.augs(img)
        if not isinstance(aug_x, dict):
            flip = False
            flip_direction = 'horizontal'
        else:
            chosen_aug_idx = torch.randint(0,len(aug_x['img']),(1,)).item()
            flip = aug_x['flip'][chosen_aug_idx]
            flip_direction = aug_x['flip_direction'][chosen_aug_idx]
            aug_x = aug_x['img'][chosen_aug_idx]
        
        return aug_x, flip, flip_direction
    
    def adjust_aug(self, cur_pred, orig_shape, flip, flip_direction):
        if flip:
            # TODO: make sure that the right dimension is flipped
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                cur_pred = torch.flip(cur_pred, [3])
            elif flip_direction == 'vertical':
                cur_pred = torch.flip(cur_pred, [2])
        cur_pred = F.interpolate(cur_pred, orig_shape, mode="bilinear", align_corners=False)
        return cur_pred

    def ignore_upper_part(self, pred):
        pred_depth = pred.clone()
        pred_depth[:, :, :int(0.3*pred_depth.shape[2]), :] = self.loss_func.ignore_index
        return pred_depth

    def depth_to_inverse(self, depth):
        clamped_depth = self.clamp_pred(depth)
        return torch.reciprocal(clamped_depth)
    
    def scale_inv_loss(self, pred, target):
        mask = target > 1.0
        d = torch.log(pred[mask]) - torch.log(target[mask])
        # scale invariant
        variance_focus = 0.85
        return torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)
        # self.effective_rank_holdout.update(inputs["color_uncrop", 0, 0] - self.opt.mean / self.opt.std)
        
        input_img = (inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std
        
        with torch.no_grad():
            features = self.models_ema["encoder"](input_img)
            pred_depth = self.models_ema["depth"](features)
            # inv_mask = pred_depth < self.opt.MIN_DEPTH
            pred_depth = torch.clamp(pred_depth, min=self.opt.MIN_DEPTH, max=self.opt.MAX_DEPTH)
            pred_depth = depth_to_disp(pred_depth, self.opt.MIN_DEPTH, self.opt.MAX_DEPTH)
            # pred_depth[inv_mask] = self.loss_func.ignore_index 
        
        aug_x_1, *aug_adjust_vars_1 = self.get_aug_image(input_img)
        # aug_x_2, *aug_adjust_vars_2 = self.get_aug_image(input_img)
        
        feats_1 = self.models["encoder"](aug_x_1)
        pred_depth_1 = self.models["depth"](feats_1)
        pred_depth_1 = torch.clamp(pred_depth_1, min=self.opt.MIN_DEPTH, max=self.opt.MAX_DEPTH)
        pred_depth_1 = depth_to_disp(pred_depth_1, self.opt.MIN_DEPTH, self.opt.MAX_DEPTH)
        # p1 = self.predictor(pred_depth_1)
        
        # with torch.no_grad():
        #     feats_2 = self.models["encoder"](aug_x_2)
        #     pred_depth_2 = self.models["depth"](feats_2)
        #     # pred_depth_2 = depth_to_disp(pred_depth_2, self.opt.MIN_DEPTH, self.opt.MAX_DEPTH)
        #     # p2 = self.predictor(pred_depth_2)
            

            # feats = self.models_source["encoder"](input_img)
            # pred_depth_source = self.models_source["depth"](feats)
            # pred_depth_source = self.clamp_pred(pred_depth_source)
        
        # feats_1 = self.models["encoder"](aug_x_1)
        # pred_depth_1 = self.models["depth"](feats_1)
        # adj_pred_depth_1 = self.adjust_aug(pred_depth_1, input_img.shape[2:], *aug_adjust_vars_1)

        adj_pred_depth_1 = self.adjust_aug(pred_depth_1, input_img.shape[2:], *aug_adjust_vars_1)
        # adj_pred_depth_2 = self.adjust_aug(pred_depth_2, input_img.shape[2:], *aug_adjust_vars_2)
        # adj_p1 = self.adjust_aug(p1, input_img.shape[2:], *aug_adjust_vars_1)
        # adj_p2 = self.adjust_aug(p2, input_img.shape[2:], *aug_adjust_vars_2)

        loss = 0.0
        # _, loss_1 = self.loss_func(adj_p1, self.clamp_pred(adj_pred_depth_2).detach())
        # _, loss_2 = self.loss_func(adj_p2, self.clamp_pred(adj_pred_depth_1).detach()) 
        _, loss_1 = self.loss_func(adj_pred_depth_1, self.ignore_upper_part(pred_depth).detach())
        
        loss += (loss_1 * 1e5)
        # loss += loss_2
        
        # with torch.no_grad():
        #     features = self.models_source["encoder"](input_img)
        #     src_pred = self.models_source["depth"](features)
        #     src_pred = torch.clamp(src_pred, min=self.opt.MIN_DEPTH, max=self.opt.MAX_DEPTH)
        #     src_pred = depth_to_disp(src_pred, self.opt.MIN_DEPTH, self.opt.MAX_DEPTH)
        # _, src_reg_loss = self.loss_func(adj_pred_depth_1, self.ignore_upper_part(src_pred))

        # loss += self.opt.alpha * src_reg_loss + (1-self.opt.alpha) * loss_1
        
        # median_loss = ((torch.median(p1) / torch.median(pred_depth_source)) - 1) ** 2
        # loss += self.opt.alpha * median_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for model, model_ema in zip(self.models.values(), self.models_ema.values()):
            update_ema_variables(model, model_ema, 0.999)

        with torch.no_grad():
            features = self.models_ema["encoder"](input_img)
            depth_out = self.models_ema["depth"](features)
            # features = self.models["encoder"](input_img)
            # depth_out = self.models["depth"](features)
            depth_out_spectral_metrics = compute_depth_out_spectral_diagnostics(depth_out)
            depth_out_buffer_metrics = self.depth_out_buffer_diagnostics.update(depth_out)
            feat_buffer_metrics = self.feature_buffer_diagnostics.update(features[-1])
            weights_l2_norm = torch.sqrt(sum(
                param.detach().pow(2).sum()
                for param in self.optimizer.param_groups[0]["params"]
            )).item()

        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=True))
        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=False))

        # error_teacher_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], aug_depth, median_scaling=True))
        # error_teacher = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], aug_depth, median_scaling=False))
        
        # self.avg_depth_errors.register_data(inputs['depth_gt_uncrop'], depth_out)
        # self.avg_depth_errors_median_scaling.register_data(inputs['depth_gt_uncrop'], depth_out)

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()

        # for idx, term in enumerate(error_teacher):
        #     error_teacher[idx] = term.detach().cpu().numpy()
        # for idx, term in enumerate(error_teacher_local):
        #     error_teacher_local[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_out.detach()

        losses = {
            'loss': loss.detach().item(),
            'diag/weights_l2_norm': weights_l2_norm,
        }
        losses.update(depth_out_spectral_metrics)
        losses.update(depth_out_buffer_metrics)
        losses.update(feat_buffer_metrics)
        depth_gt = F.interpolate(inputs['depth_gt_uncrop'], [depth_out.shape[2], depth_out.shape[3]], mode="bilinear", align_corners=False)
        losses.update(
            {
                "diag/grad_mag_dir_error": compute_gradient_magnitude_direction_error(
                    input_img,
                    depth_out),
                "diag/grad_chamf_dist": compute_image_gradient_chamfer_distance(
                    input_img,
                    depth_out),
                "diag/depth_out_tv": compute_total_variation(depth_out),
                "diag/depth_out_tv_iso": compute_total_variation(depth_out, isotropic=True),
                "diag/depth_out_tv_error": compute_tv_error(depth_out, depth_gt),
                "diag/depth_out_residual_tv": compute_residual_total_variation(
                    depth_out,
                    depth_gt),
            }
        )
        
        if self.step == 0 or self.step % 50 == 0:
            eff_rank = self.er.get_effective_rank(self.models_ema["encoder"], return_details=True)
            # eff_rank = self.er.get_effective_rank(self.models["encoder"], return_details=True)
            losses.update({
                "diag/effective_rank_all": eff_rank['all'],
                # "diag/effective_rank_last": eff_rank['last']
            })
        
        metrics = {
            'error': error,
            'error_local': error_local,
            # 'error_teacher': error_teacher,
            # 'error_teacher_local': error_teacher_local,
        }
        self.step += 1
        self.steps_since_reset += 1
        
        # if self.steps_since_reset >= self.reset_every:
        #     for name, model in self.models.items():
        #         model.load_state_dict(self.initial_model_states[name])
        #     for name, model in self.models_ema.items():
        #         model.load_state_dict(self.initial_model_states[name])

        #     self.optimizer.load_state_dict(self.initial_optimizer_state)
        #     self.steps_since_reset = 0

        
        return outputs, metrics, losses
