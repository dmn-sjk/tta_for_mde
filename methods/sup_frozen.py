
from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP, AvgDepthErrorsPerDepthRange

import torch
from layers import disp_to_depth, compute_depth_errors_adadepth, compute_depth_errors
from networks import get_supervised_models
from utils.diagnostics import (
    DepthOutBufferDiagnostics,
    compute_depth_out_spectral_diagnostics,
    FeauturesBufferDiagnostics
)
from utils.effective_rank import EffectiveRank, EffectiveRankHoldout
import os


@register_method(name='sup_frozen')
class SupFrozen(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.models = get_supervised_models(self.opt)
        
        for m in self.models.values():
            m.eval()
        
        # parameters_to_train = []
        # parameters_to_train += list(self.models["encoder"].parameters())
        # parameters_to_train += list(self.models["depth"].parameters())
        # self.optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)
        
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
        self.er = EffectiveRank(holdout_path=self.effective_rank_holdout_path, device=self.opt.device)
        self.step = 0
 
    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)

        with torch.no_grad():
            features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            depth_out = self.models["depth"](features)
            
            depth_out_spectral_metrics = compute_depth_out_spectral_diagnostics(depth_out)
            depth_out_buffer_metrics = self.depth_out_buffer_diagnostics.update(depth_out)
            feat_buffer_metrics = self.feature_buffer_diagnostics.update(features[-1])

        # pred_dict = {('depth', 0): depth_out}
        # pred_dict.update(self.predict_poses(inputs, features, self.models))
        # self.generate_images_pred(inputs, pred_dict, disp_input=False)
        # # # mask the upper part of the prediction for loss computation
        # # mask = torch.zeros_like(depth_out.squeeze(1))
        # # mask[:, depth_out.shape[2]//2:, :] = 1 
        # unsup_losses = self.compute_losses_unsup(inputs, pred_dict, disp_input=False, mask=None)
        # self.optimizer.zero_grad()
        # unsup_losses["loss"].backward()
        # self.optimizer.step()
        
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

        losses = {}
        losses.update(depth_out_spectral_metrics)
        losses.update(depth_out_buffer_metrics)
        losses.update(feat_buffer_metrics)
        
        if self.step == 0 or self.step % 50 == 0:
            eff_rank = self.er.get_effective_rank(self.models["encoder"], return_details=True)
            losses.update({
                "diag/effective_rank_all": eff_rank['all'],
                # "diag/effective_rank_last": eff_rank['last']
            })
        
        metrics = {
            'error': error,
            'error_local': error_local,
        }
        
        self.step += 1
        
        return outputs, metrics, losses
