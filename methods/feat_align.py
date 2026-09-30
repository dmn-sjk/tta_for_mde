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
from networks.hog_layer import HOGLayerC
from layers import AvgDepthErrorsPerDepthRange
import os


@register_method(name='feat_align')
class FeatAlign(BaseAdaptation):
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

        parameters_to_train = []
        parameters_to_train += list(self.models["encoder"].parameters())
        # parameters_to_train += list(self.models["depth"].parameters())
        self.optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)

        self.proto = torch.load('data/supervised_prototypes.pth')
        for i, p in enumerate(self.proto):
            self.proto[i]['mean'] = self.proto[i]['mean'].to(self.opt.device)
            self.proto[i]['cov'] = self.proto[i]['cov'].to(self.opt.device)
 
    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)
        
        input_img = (inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std
        B, _, H, W = input_img.shape

        feats = self.models["encoder"](input_img)
        # outputs = self.models["depth"](feats)
    
        loss = 0
        for i, f in enumerate(feats):
            # proto = F.interpolate(self.proto[i]['mean'].unsqueeze(0), f.shape[-2:], mode="bilinear", align_corners=False)
            # loss += torch.nn.functional.mse_loss(f, proto)
            curr_fs = f.view(f.shape[1], -1)
            curr_mean = torch.mean(curr_fs, dim=-1)
            curr_cov = torch.cov(curr_fs)
            loss += torch.linalg.norm((curr_mean - self.proto[i]['mean']).flatten(), ord=2) + \
                torch.linalg.norm((curr_cov - self.proto[i]['cov']), ord='fro')
        loss /= len(feats)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        with torch.no_grad():
            feats = self.models["encoder"](input_img)
            depth_out = self.models["depth"](feats)

        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=True))
        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=False))

        self.avg_depth_errors.register_data(inputs['depth_gt_uncrop'], depth_out)
        self.avg_depth_errors_median_scaling.register_data(inputs['depth_gt_uncrop'], depth_out)

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()

        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_out
        # outputs['depth_model_pred'] = pred_depth
        losses = {}
        losses['loss_proto'] = loss.detach().cpu().numpy()
        # losses['loss_ssl'] = unsup_losses['loss'].detach().cpu().numpy()
        # losses['loss_recon'] = loss_recon.detach().cpu().numpy()
        # losses['loss_reg'] = loss_reg.detach().cpu().numpy()
        
        metrics = {
            'error': error,
            'error_local': error_local,
        }
        
        return outputs, metrics, losses
