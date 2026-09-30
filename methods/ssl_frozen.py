from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
from layers import disp_to_depth, compute_depth_errors_adadepth
from networks import get_self_supervised_models


@register_method(name='ssl_frozen')
class SSLFrozen(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.models = get_self_supervised_models(self.opt)
        
        for m in self.models.values():
            m.eval()

    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)

        with torch.no_grad():
            reg_features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            reg_outputs = self.models["depth"](reg_features)
            reg_outputs.update(self.predict_poses(inputs, reg_features, self.models))
            _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            self.generate_images_pred(inputs, reg_outputs)
            unsup_losses = self.compute_losses_unsup(inputs, reg_outputs)
        
        error_unsup_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=True))
        error_unsup = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=False))

        for idx, term in enumerate(error_unsup):
            error_unsup[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup_local):
            error_unsup_local[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = reg_depth_unsup

        losses = {}

        for loss_type, loss_val in unsup_losses.items():
            losses['unsup_' + loss_type] = loss_val.detach().cpu()
        
        metrics = {
            'error': error_unsup,
            'error_local': error_unsup_local,
        }
        
        return outputs, metrics, losses
