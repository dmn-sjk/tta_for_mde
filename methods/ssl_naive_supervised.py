
from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
from layers import disp_to_depth, depth_to_disp, compute_depth_errors_adadepth
from networks import get_supervised_models, get_self_supervised_models


@register_method(name='ssl_naive_supervised')
class SSLNaiveSupervised(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.models = get_supervised_models(self.opt)
        ssl_models = get_self_supervised_models(self.opt)
        self.models["pose_encoder"] = ssl_models["pose_encoder"]
        self.models["pose"] = ssl_models["pose"]
        
        parameters_to_train = []
        parameters_to_train += list(self.models["encoder"].parameters())   
        parameters_to_train += list(self.models["depth"].parameters())

        parameters_to_train += list(self.models["pose_encoder"].parameters())
        parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)
        
    def process_batch(self, inputs):
        # Update supervised model with self-supervised loss using GT ego motion
        inputs_to_device(inputs, self.opt.device)

        for m in self.models.values():
            m.train()

        # adapt supervised model with self-supervised loss using GT ego motion
        features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
        depth = self.models["depth"](features)
        # TODO: disparity is only used in the disparity smoothness loss, might want to remove the loss
        disp = depth_to_disp(depth, self.opt.min_depth, self.opt.max_depth)
        outputs = {
            ('depth', 0): depth, 
            ("disp", 0): disp
            }

        outputs.update(self.predict_poses(inputs, None, self.models))
        self.generate_images_pred(inputs, outputs, disp_input=False)

        unsup_losses = self.compute_losses_unsup(inputs, outputs)
        self.model_optimizer.zero_grad()
        unsup_losses["loss"].backward()
        self.model_optimizer.step()

        for m in self.models.values():
            m.eval()

        with torch.no_grad():
            features_uncrop = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            depth_uncrop = self.models["depth"](features_uncrop)

        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop))
        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop, median_scaling=True))

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()


        outputs = {}
        outputs['depth'] = depth_uncrop

        losses = {}
        for i, metric in enumerate(DEPTH_METRIC_NAMES):
            losses[metric] = error[i]
        for i, metric in enumerate(DEPTH_METRIC_NAMES_LOCAL):
            losses[metric] = error_local[i]
        for loss_type, loss_val in unsup_losses.items():
            losses['unsup_' + loss_type] = loss_val
            
        
        metrics = {
            'error': error,
            'error_local': error_local,
        }

        return outputs, metrics, losses