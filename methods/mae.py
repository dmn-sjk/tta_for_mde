from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
from layers import disp_to_depth, compute_depth_errors_adadepth, compute_depth_errors, BackprojectDepth
from networks import get_supervised_models, get_self_supervised_models
from utils.svdp_augs import SVDPMultiScaleFlipAug, Resize, RandomFlip
from utils.losses import DepthLoss
from layers import update_ema_variables, SingleStageReconHead, MultiStageReconHead
from utils.utils import save_tensor_as_image
import torch.nn.functional as F
import copy
import torch.nn as nn
import os


@register_method(name='mae')
class MAE(BaseAdaptation):
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

        head_dim = 192 # embed_dim of swin transformer
        self.patch_size = 4
        self.mask_ratio = self.opt.mask_ratio

        # mask_token
        mask_token_dim = (1, 1, head_dim)
        self.mask_token = nn.Parameter(torch.zeros(*mask_token_dim), 
                                       requires_grad=False)
        self.mask_token = nn.DataParallel(self.mask_token) # make parallel
        self.mask_token.cuda()
        
        if self.opt.masking == "simple":
            print("Using simple masking!")
            self.mask_fn = simple_masking
        elif self.opt.masking == "shuffle":
            print("Using shuffle masking!")
            self.mask_fn = shuffle_masking
        else:
            raise NotImplementedError()
        
        # create ema models
        self.models_ema = copy.deepcopy(self.models)
        for model in self.models_ema.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                param.detach_()  

        parameters_to_train = []
        parameters_to_train += list(self.models["encoder"].parameters())
        parameters_to_train += list(self.models["depth"].parameters())
        # parameters_to_train += list(self.mask_token.parameters())
        self.optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)
 
        # self.svdp_augs = SVDPMultiScaleFlipAug(
        #     img_scale=(self.opt.height, self.opt.width), 
        #     img_ratios=[
        #         0.5, 
        #         0.75,
        #         1.0, 
        #         1.25, 
        #         1.5, 
        #         1.75, 
        #         2.0
        #         ],
        #     flip=True,
        #     patch_size=self.opt.patch_size,
        #     transforms=[
        #         Resize(keep_ratio=True),
        #         RandomFlip(),
        #         # dict(type='Normalize', **img_norm_cfg), # assume the images are already normalized form dataloader
        #     ])

        self.loss = DepthLoss(loss='l1')

        self.idx = 0
        
    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)
        
        input_img = (inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std
        B, _, H, W = input_img.shape
        orig_patch_nums = (H//self.patch_size, W//self.patch_size)
        
        with torch.no_grad():
            ref_features = self.models_ema["encoder"](input_img)
            pred_depth = self.models_ema["depth"](ref_features)
            
        mask_chosed = (torch.rand(orig_patch_nums).flatten().unsqueeze(0) < self.mask_ratio).to(torch.float32)  
        
        feats = self.models["encoder"](input_img, self.mask_fn, self.mask_token, mask_chosed)
        outputs = self.models["depth"](feats)
    
        pred_depth = torch.clamp(pred_depth, min=self.opt.MIN_DEPTH, max=self.opt.MAX_DEPTH)
        pred_depth[:, :, :int(0.3*pred_depth.shape[2]), :] = self.loss.ignore_index

        loss_mask, loss = self.loss(outputs, pred_depth)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        for model, model_ema in zip(self.models.values(), self.models_ema.values()):
            update_ema_variables(model, model_ema, 0.999)
        
        # for model, anchor in zip(self.models.values(), self.anchors.values()): 
        #     for nm, m in model.named_modules():
        #         for npp, p in m.named_parameters():
        #             if npp in ['weight', 'bias'] and p.requires_grad:
        #                 mask = (torch.rand(p.shape)<0.01).float().cuda()
        #                 with torch.no_grad():
        #                     p.data = anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)

        with torch.no_grad():
            feats = self.models_ema["encoder"](input_img)
            depth_out = self.models_ema["depth"](feats)

        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=True))
        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=False))
        
        error_teacher_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], outputs, median_scaling=True))
        error_teacher = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], outputs, median_scaling=False))

        self.avg_depth_errors.register_data(inputs['depth_gt_uncrop'], depth_out)
        self.avg_depth_errors_median_scaling.register_data(inputs['depth_gt_uncrop'], depth_out)

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()

        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()
        
        for idx, term in enumerate(error_teacher):
            error_teacher[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher_local):
            error_teacher_local[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_out
        outputs['depth_model_pred'] = pred_depth
        losses = {}
        losses['loss'] = loss.detach().cpu().numpy()
        # losses['loss_ssl'] = unsup_losses['loss'].detach().cpu().numpy()
        # losses['loss_recon'] = loss_recon.detach().cpu().numpy()
        # losses['loss_reg'] = loss_reg.detach().cpu().numpy()
        
        metrics = {
            'error': error,
            'error_local': error_local,
            'error_teacher': error_teacher,
            'error_teacher_local': error_teacher_local,
        }
        
        return outputs, metrics, losses


def simple_masking(x: torch.Tensor, mask_token: torch.nn.Parameter, 
                   unmask: torch.Tensor):
    """
    x: (B, num_patches, dim_feats),
    mask_token: (1, 1, dim_feats),
    unmask: (1, num_patches)
    """
    B, N, dim_feats = x.shape
    device = x.device
    
    mask_tokens = mask_token.module.expand(B, N, -1).to(device)
    unmask = unmask.unsqueeze(2).to(device) #mask_chosen
    return x * (1-unmask) + mask_tokens * unmask

def shuffle_masking(x: torch.Tensor, mask_token: torch.nn.Parameter, 
                   unmask: torch.Tensor):
    """
    x: (B, num_patches, dim_feats),
    mask_token: (1, 1, dim_feats),
    unmask: (1, num_patches)
    """
    B, N, dim_feats = x.shape
    device = x.device

    mask = unmask.to(torch.bool)
    perm_indices = torch.randperm(dim_feats, device=device)
    # Apply permutation only where mask == 1
    # Use masked indexing to shuffle
    x[mask] = x[mask][:, perm_indices]
    return x
 