from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
from layers import disp_to_depth, compute_depth_errors_adadepth, compute_depth_errors
from networks import get_supervised_models, get_self_supervised_models
from utils.svdp_augs import SVDPMultiScaleFlipAug, Resize, RandomFlip
from utils.losses import DepthLoss
from layers import update_ema_variables
from utils.utils import save_tensor_as_image
import torch.nn.functional as F
import copy


@register_method(name='cotta')
class CoTTA(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        if self.opt.model_type == "supervised":
            self.models = get_supervised_models(self.opt)
        else:
            # self.models = get_self_supervised_models(self.opt)
            # TODO: loss calculation is different for self-supervised models
            raise NotImplementedError("Self-supervised models not implemented yet")

        for m in self.models.values():
            m.eval()
        
        # create ema models
        # ema models for regularisation of global scale
        self.models_ema = copy.deepcopy(self.models)
        for model in self.models_ema.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                param.detach_()
 
        parameters_to_train = []
        parameters_to_train += list(self.models["encoder"].parameters())
        parameters_to_train += list(self.models["depth"].parameters())
        self.optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)
 
        self.svdp_augs = SVDPMultiScaleFlipAug(
            img_scale=(self.opt.height, self.opt.width), 
            img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            flip=True,
            patch_size=self.opt.patch_size,
            transforms=[
                Resize(keep_ratio=True),
                RandomFlip(),
                # dict(type='Normalize', **img_norm_cfg), # assume the images are already normalized form dataloader
            ])
        
        self.loss = DepthLoss(loss='mse')

        self.anchors = {k: copy.deepcopy(val.state_dict()) for k, val in self.models.items()}
        
        self.frozen_models = copy.deepcopy(self.models)
        for model in self.frozen_models.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                param.detach_()

    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)
        
        input_img = (inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std

        features = self.models["encoder"](input_img)
        pred_depth = self.models["depth"](features)


        # pred_depth = F.interpolate(pred_depth, inputs['depth_gt_uncrop'].shape[-2:], mode="bilinear", align_corners=False)
        # # gt_depth = F.interpolate(inputs['depth_gt_uncrop'], pred_depth.shape[-2:], mode="bilinear", align_corners=False)
        # gt_depth = inputs['depth_gt_uncrop']
        # invalid_gt_mask = torch.logical_and(gt_depth < self.opt.MIN_DEPTH, gt_depth > self.opt.MAX_DEPTH)
        # gt_depth[invalid_gt_mask] = self.loss.ignore_index
        # # aug_depth = gt_depth

        aug_x = self.svdp_augs({'img': input_img})
        aug_depth = torch.zeros_like(pred_depth)
        for x, flip, flip_direction in zip(aug_x['img'], aug_x['flip'], aug_x['flip_direction']):
            with torch.no_grad():
                feats = self.models_ema["encoder"](x)
                cur_pred = self.models_ema["depth"](feats)
 
            if flip:
                # TODO: make sure that the right dimension is flipped
                assert flip_direction in ['horizontal', 'vertical']
                if flip_direction == 'horizontal':
                    cur_pred = torch.flip(cur_pred, [3])
                elif flip_direction == 'vertical':
                    cur_pred = torch.flip(cur_pred, [2])
            cur_pred = F.interpolate(cur_pred, pred_depth.shape[2:], mode="bilinear", align_corners=False)
            
            aug_depth = aug_depth + cur_pred
        # average prediction
        aug_depth = aug_depth / len(aug_x['img'])
        
        # filter out pseudo-labels that deviate too much from GT
        # diff_threshold = self.opt.alpha  # adjust threshold as needed
        # depth_diff = torch.abs(aug_depth - gt_depth)
        # invalid_pseudo_mask = depth_diff > diff_threshold
        # aug_depth[invalid_pseudo_mask] = self.loss.ignore_index
        # self.adapt_instance.writer.add_scalar('num_valid_pseudo_depth_pixels', 
        #                        (aug_depth != self.loss.ignore_index).sum(), 
        #                        self.adapt_instance.step)



        if self.opt.n_bins > 0:
            aug_depth = self.discretize_depth_log(aug_depth, n_bins=self.opt.n_bins)

        loss = self.loss(pred_depth, aug_depth)[-1]

        # if self.opt.alpha > 0:
        #     with torch.no_grad():
        #         features = self.frozen_models["encoder"](input_img)
        #         frozen_depth = self.frozen_models["depth"](features)
               
        #     # loss_frozen = self.loss(pred_depth, frozen_depth).mean()
            
        #     lambda1 = 0.1
        #     lambda2 = 0.1
        #     mu_0 = frozen_depth.mean()
        #     sigma_0 = frozen_depth.std()
        #     mu_current = pred_depth.mean()
        #     sigma_current = pred_depth.std()
        #     loss_frozen = lambda1 * (mu_current - mu_0)**2 + lambda2 * (sigma_current - sigma_0)**2
            
        #     loss = (1-self.opt.alpha) * loss + self.opt.alpha * loss_frozen

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for model, model_ema in zip(self.models.values(), self.models_ema.values()):
            update_ema_variables(model, model_ema, 0.999)
       
        for model, anchor in zip(self.models.values(), self.anchors.values()): 
            for nm, m in model.named_modules():
                for npp, p in m.named_parameters():
                    if npp in ['weight', 'bias'] and p.requires_grad:
                        mask = (torch.rand(p.shape)<0.01).float().cuda()
                        with torch.no_grad():
                            p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)
            
        with torch.no_grad():
            feats = self.models_ema["encoder"](input_img)
            depth_out = self.models_ema["depth"](feats)
            # depth_out = F.interpolate(depth_out, inputs['depth_gt_uncrop'].shape[-2:], mode="bilinear", align_corners=False)

        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=True))
        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=False))

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()

        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_out
        outputs['depth_model_pred'] = pred_depth
        outputs['depth_aug_pred'] = aug_depth
        losses = {}
        
        metrics = {
            'error': error,
            'error_local': error_local,
        }
        
        return outputs, metrics, losses

    def discretize_depth(self, depth_tensor, n_bins=20):
        """
        Discretize depth values in a tensor into n evenly spaced bins.
        
        Args:
            depth_tensor: torch tensor of shape [B,C,H,W] containing depth values
            n_bins: number of bins to divide the depth range into
            
        Returns:
            Torch tensor of same shape with discretized depth values
        """
        min_depth = torch.min(depth_tensor)
        max_depth = torch.max(depth_tensor)
        
        # Create bin edges
        bin_edges = torch.linspace(min_depth, max_depth, n_bins + 1, device=depth_tensor.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # For each depth value, find the closest bin center
        expanded_depth = depth_tensor.unsqueeze(-1)  # Add dimension for bin comparison
        expanded_centers = bin_centers.view(1, 1, 1, 1, -1)  # Shape for broadcasting
        
        # Calculate absolute difference to each bin center
        abs_diff = torch.abs(expanded_depth - expanded_centers)
        
        # Find index of minimum difference (closest bin)
        closest_bin_idx = torch.argmin(abs_diff, dim=-1)
        
        # Get the value of the closest bin center
        discretized_depth = bin_centers[closest_bin_idx]
        
        return discretized_depth
    
    def discretize_depth_log(self, depth_tensor, n_bins):
        """
        Discretize depth values in a tensor into n logarithmically spaced bins,
        making bins more granular at lower depth values.
        
        Args:
            depth_tensor: torch tensor of shape [B,C,H,W] containing depth values
            n_bins: number of bins to divide the depth range into
            
        Returns:
            Torch tensor of same shape with discretized depth values
        """
        eps = 1e-6  # small epsilon to avoid log(0)
        min_depth = torch.min(depth_tensor).clamp(min=eps)
        max_depth = torch.max(depth_tensor).clamp(min=min_depth + eps)

        # Create logarithmically spaced bin edges
        bin_edges = torch.logspace(
            torch.log2(min_depth), torch.log2(max_depth), n_bins + 1, base=2, device=depth_tensor.device
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # For each depth value, find the closest bin center
        expanded_depth = depth_tensor.unsqueeze(-1)  # Add dimension for bin comparison
        expanded_centers = bin_centers.view(1, 1, 1, 1, -1)  # Shape for broadcasting

        # Calculate absolute difference to each bin center
        abs_diff = torch.abs(expanded_depth - expanded_centers)

        # Find index of minimum difference (closest bin)
        closest_bin_idx = torch.argmin(abs_diff, dim=-1)

        # Get the value of the closest bin center
        discretized_depth = bin_centers[closest_bin_idx]

        return discretized_depth