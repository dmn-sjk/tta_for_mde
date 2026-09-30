from layers import disp_to_depth, update_ema_variables, compute_depth_errors_adadepth, \
    DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP, AvgDepthErrorsPerDepthRange
from methods.base import BaseAdaptation
from utils.utils import inputs_to_device
from networks import get_supervised_models, get_self_supervised_models
from methods import register_method

import torch
import copy
import torch.optim as optim
from utils.effective_rank import EffectiveRank, EffectiveRankHoldout
from utils.diagnostics import (
    DepthOutBufferDiagnostics,
    compute_depth_out_spectral_diagnostics,
)
import os


@register_method(name='adadepth')
class AdaDepth(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)

        self.models = get_supervised_models(self.opt)
        parameters_to_train = []
        parameters_to_train += list(self.models["encoder"].parameters())
        parameters_to_train += list(self.models["depth"].parameters())

        # ema models for regularisation of global scale
        self.models_ema = copy.deepcopy(self.models)
        for m in self.models_ema.values():
            m.eval()

        # reference models for improvements calculation
        self.models_ref = copy.deepcopy(self.models)
        self.models_ref_state = {}
        for key, value in self.models_ref.items():
            self.models_ref_state[key] = copy.deepcopy(self.models[key].state_dict())
            value.eval()
    
        self.model_optimizer = optim.Adam(parameters_to_train, self.opt.learning_rate)

        # models for regularisation, self-supervised
        print("loading regularisation model from " + self.opt.ssl_model_path)
        # create the reg model
        self.reg_models = get_self_supervised_models(self.opt)
        reg_parameters_to_train = []
        reg_parameters_to_train += list(self.reg_models["encoder"].parameters())
        reg_parameters_to_train += list(self.reg_models["depth"].parameters())
        reg_parameters_to_train += list(self.reg_models["pose_encoder"].parameters())
        reg_parameters_to_train += list(self.reg_models["pose"].parameters())

        if self.opt.adaptation_method == 'adadepth':
            self.reg_models_ref = copy.deepcopy(self.reg_models)
            for m in self.reg_models_ref.values():
                m.eval()
        
        self.reg_model_optimizer = optim.Adam(reg_parameters_to_train, self.opt.learning_rate)
        
        self.depth_out_buffer_diagnostics = DepthOutBufferDiagnostics(
            maxlen=getattr(self.opt, "depth_out_analysis_buffer_size", 50)
        )
        self.effective_rank_holdout_path = os.path.join(
            "/net/pr2/projects/plgrid/plgguseby/dmns/ada-depth", "effective_rank_holdout",
            f"effective_rank_holdout_{self.opt.dataset}.pt",
        )
        self.er = EffectiveRank(holdout_path=self.effective_rank_holdout_path, device=self.opt.device)
        self.step = 0

    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)
        
        # first update the self-supervised models
        for m in self.reg_models.values():
            m.train()

        reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
        reg_outputs = self.reg_models["depth"](reg_features)
        reg_outputs.update(self.predict_poses(inputs, reg_features, self.reg_models))
        _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
        self.generate_images_pred(inputs, reg_outputs)
        unsup_losses = self.compute_losses_unsup(inputs, reg_outputs)
        self.reg_model_optimizer.zero_grad()
        unsup_losses["loss"].backward()
        self.reg_model_optimizer.step()

        for m in self.reg_models.values():
            m.eval()

        for m in self.models.values():
            m.train()

        # run adaptation with pseudo labels
        features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
        depth = self.models["depth"](features)
        with torch.no_grad():
            reg_features = self.reg_models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            reg_outputs = self.reg_models["depth"](reg_features)
            reg_outputs.update(self.predict_poses(inputs, reg_features, self.reg_models))
            _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            # reference models
            features_uncrop_ref = self.models_ref["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            depth_uncrop_ref = self.models_ref["depth"](features_uncrop_ref)

            features_uncrop_ema = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            depth_uncrop_ema = self.models_ema["depth"](features_uncrop_ema)

            reg_features_ref = self.reg_models_ref["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            reg_outputs_ref = self.reg_models_ref["depth"](reg_features_ref)
            _, depth_unsup_ref = disp_to_depth(reg_outputs_ref[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

            pseudo_depth_sup, pseudo_depth_unsup = self.augment_pseudo(inputs,
                                                                       depth_uncrop_ref,
                                                                       reg_depth_unsup,
                                                                       depth_uncrop_ema
                                                                       )

        loss_ada = self.compute_losses(depth, pseudo_depth_sup, pseudo_depth_unsup)
        self.model_optimizer.zero_grad()
        loss_ada.backward()
        self.model_optimizer.step()

        # update ema models
        update_ema_variables(self.models["encoder"],
                             self.models_ema["encoder"],
                             0.99)
        update_ema_variables(self.models["depth"],
                             self.models_ema["depth"],
                             0.99)

        for m in self.models.values():
            m.eval()

        with torch.no_grad():
            features_uncrop = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            depth_uncrop = self.models["depth"](features_uncrop)
            depth_out_spectral_metrics = compute_depth_out_spectral_diagnostics(depth_uncrop)
            depth_out_buffer_metrics = self.depth_out_buffer_diagnostics.update(depth_uncrop)

            features_uncrop_ema = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            depth_uncrop_ema = self.models_ema["depth"](features_uncrop_ema)

        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop))
        error_teacher = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop_ema))
        error_ref = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop_ref))
        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop, median_scaling=True))
        error_teacher_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop_ema, median_scaling=True))
        error_local_ref = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_uncrop_ref, median_scaling=True))

        error_unsup = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=True))
        error_unsup_ref = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_unsup_ref, median_scaling=True))

        self.avg_depth_errors.register_data(inputs['depth_gt_uncrop'], depth_uncrop)
        self.avg_depth_errors_median_scaling.register_data(inputs['depth_gt_uncrop'], depth_uncrop)

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher):
            error_teacher[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_teacher_local):
            error_teacher_local[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup):
            error_unsup[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_ref):
            error_ref[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local_ref):
            error_local_ref[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_unsup_ref):
            error_unsup_ref[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_uncrop
        outputs['pseudo_depth_sup'] = pseudo_depth_sup
        outputs['pseudo_depth_unsup'] = pseudo_depth_unsup

        losses = {}
        losses['loss'] = loss_ada.detach().cpu().numpy()
        for i, metric in enumerate(DEPTH_METRIC_NAMES):
            losses[metric] = error[i]
        for i, metric in enumerate(DEPTH_METRIC_NAMES_LOCAL):
            losses[metric] = error_local[i]
        for loss_type, loss_val in unsup_losses.items():
            losses['unsup_' + loss_type] = loss_val
        
        losses.update(depth_out_spectral_metrics)
        losses.update(depth_out_buffer_metrics)
        
        if self.step == 0 or self.step % 50 == 0:
            eff_rank = self.er.get_effective_rank(self.models_ema["encoder"], return_details=True)
            losses.update({
                "diag/effective_rank_all": eff_rank['all'],
                # "diag/effective_rank_last": eff_rank['last']
            })
            
        
        metrics = {
            'error': error,
            'error_local': error_local,
            'error_unsup': error_unsup,
            'error_ref': error_ref,
            'error_local_ref': error_local_ref,
            'error_unsup_ref': error_unsup_ref,
            'error_teacher': error_teacher,
            'error_teacher_local': error_teacher_local,
        }
        
        self.step += 1

        return outputs, metrics, losses

    def augment_pseudo(self, inputs, reg_depth_sup, reg_depth_unsup, depth_uncrop_ref):
        # reg depth sup is merely the current depth; difference is train/eval modes
        # generate mask by thresholding (output)
        # ema model
        mask0 = depth_uncrop_ref > self.opt.MIN_DEPTH
        mask0[:, :, :int(0.3*depth_uncrop_ref.shape[2]), :] = False
        scale_factor = torch.median(depth_uncrop_ref[mask0]) / torch.median(reg_depth_unsup[mask0])
        if self.opt.dataset == "kitti":
            gt_height, gt_width = depth_uncrop_ref.shape[2], depth_uncrop_ref.shape[3]
            # garg/eigen crop
            crop_mask = torch.zeros_like(mask0)
            crop_mask[:, :, int(0.40810811*gt_height):int(0.99189189*gt_height),
                      int(0.03594771*gt_width):int(0.96405229*gt_width)] = 1
            mask0 = crop_mask

        # reg depth unsup, consistency with ema model
        mask = (((depth_uncrop_ref - reg_depth_unsup * scale_factor) ** 2) / depth_uncrop_ref) < self.opt.thres

        reg_depth_unsup = torch.mul(mask.float(), reg_depth_unsup) * scale_factor
        reg_depth_unsup = torch.clamp(reg_depth_unsup, min=self.opt.MIN_DEPTH, max=self.opt.MAX_DEPTH)

        # # reg depth sup
        mask = (((reg_depth_sup - depth_uncrop_ref) ** 2) / reg_depth_sup) < self.opt.thres
        mask = mask * mask0

        reg_depth_sup = torch.mul(mask.float(), depth_uncrop_ref)
        reg_depth_sup = torch.clamp(reg_depth_sup, min=self.opt.MIN_DEPTH, max=self.opt.MAX_DEPTH)

        return reg_depth_sup, reg_depth_unsup

    def compute_losses(self, depth, pseudo_depth_sup, pseudo_depth_unsup):
        mask = pseudo_depth_sup > 1.0
        d = torch.log(depth[mask]) - torch.log(pseudo_depth_sup[mask])
        # scale invariant
        variance_focus = 0.85
        loss_sup = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

        mask = pseudo_depth_unsup > 1.0
        d = torch.log(depth[mask]) - torch.log(pseudo_depth_unsup[mask])
        # scale invariant
        variance_focus = 0.85
        loss_unsup = torch.sqrt((d ** 2).mean() - variance_focus * (d.mean() ** 2)) * 10.0

        loss = loss_sup + loss_unsup
        return loss