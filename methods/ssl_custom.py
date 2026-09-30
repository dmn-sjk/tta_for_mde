

from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
from layers import disp_to_depth, compute_depth_errors_adadepth
from networks import get_self_supervised_models
from collections import deque
import copy
from layers import update_ema_variables
from torch.nn import L1Loss


# Function to save gradients
def save_gradients(model):
    return [param.grad.clone() if param.grad is not None else None for param in model.parameters()]

# Function to apply gradients
def apply_gradients(model, gradients):
    for param, grad in zip(model.parameters(), gradients):
        if grad is not None:
            param.grad = grad.clone()


@register_method(name='ssl_custom')
class SSLCustom(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.models = get_self_supervised_models(self.opt)
        
        parameters_to_train = []
        parameters_to_train += list(self.models["encoder"].parameters())
        parameters_to_train += list(self.models["depth"].parameters())
        # parameters_to_train += list(self.models["pose_encoder"].parameters())
        # parameters_to_train += list(self.models["pose"].parameters())
        self.models['pose_encoder'].requires_grad_(False)
        self.models['pose'].requires_grad_(False)

        self.model_optimizer = torch.optim.Adam(parameters_to_train, self.opt.learning_rate)

        # num_saved_grads = 5
        
        # self.saved_gradients = {
        #     "depth": deque(maxlen=num_saved_grads),
        #     "encoder": deque(maxlen=num_saved_grads),
        #     "pose": deque(maxlen=num_saved_grads),
        #     "pose_encoder": deque(maxlen=num_saved_grads),
        # }
        # self.saved_gradients_ema = {
        #     "depth": None,
        #     "encoder": None,
        #     "pose": None,
        #     "pose_encoder": None,
        # }
        
        self.models_ema = copy.deepcopy(self.models)
        for model in self.models_ema.values():
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                param.detach_()
                
    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)

        # first update the self-supervised models
        for m in self.models.values():
            m.train()

        reg_features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
        reg_outputs = self.models["depth"](reg_features)
        reg_outputs.update(self.predict_poses(inputs, reg_features, self.models))
        _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

        self.generate_images_pred(inputs, reg_outputs)
        unsup_losses = self.compute_losses_unsup(inputs, reg_outputs)
        # self.model_optimizer.zero_grad()
        # unsup_losses["loss"].backward()
        # self.model_optimizer.step()

        for m in self.models.values():
            m.eval()
        
        with torch.no_grad():
            reg_features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            reg_outputs = self.models["depth"](reg_features)
            reg_outputs.update(self.predict_poses(inputs, reg_features, self.models))
            _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            error_unsup_bef_update = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], reg_depth_unsup, median_scaling=True))
        
            self.model_optimizer.zero_grad(set_to_none=True)
            unsup_losses["loss"].backward()
            
            # self.saved_gradients["depth"].append(save_gradients(self.models["depth"]))
            # self.saved_gradients["encoder"].append(save_gradients(self.models["encoder"]))
            # self.saved_gradients["pose"].append(save_gradients(self.models["pose"]))
            # self.saved_gradients["pose_encoder"].append(save_gradients(self.models["pose_encoder"]))
            
            
            # for key, model in self.models.items():
            #     if self.saved_gradients_ema[key] is None:
            #         self.saved_gradients_ema[key] = save_gradients(model)
            #     else:
            #         for grad, grad_ema in zip(save_gradients(model), self.saved_gradients_ema[key]):
            #             if grad is not None:
            #                 grad_ema[:] = (1-self.opt.alpha) * grad + self.opt.alpha * grad_ema
            
            
            # avg_gradients = {
            #     "depth": [],
            #     "encoder": [],
            #     "pose": [],
            #     "pose_encoder": [],
            # }
            # num_grads = len(self.saved_gradients["depth"])
            # for key, model_grads in self.saved_gradients.items():
            #     for grads in zip(*model_grads):
            #         valid_grads = [g for g in grads if g is not None]
            #         if valid_grads:
            #             avg_grad = sum(valid_grads) / num_grads
            #             avg_gradients[key].append(avg_grad)
            #         else:
            #             avg_gradients[key].append(None)
 
            # self.model_optimizer.zero_grad(set_to_none=True)
            
            # for key, model in self.models.items():
            #     # apply_gradients(model, avg_gradients[key])
            #     apply_gradients(model, self.saved_gradients_ema[key])
            
            self.model_optimizer.step()
            
            
            for model, model_ema in zip(self.models.values(), self.models_ema.values()):
                update_ema_variables(model, model_ema, 0.999)

            
            reg_features = self.models_ema["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            reg_outputs = self.models_ema["depth"](reg_features)
            _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            

            # reg_features = self.models["encoder"]((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            # reg_outputs = self.models["depth"](reg_features)
            # _, reg_depth_unsup = disp_to_depth(reg_outputs[("disp", 0)], self.opt.min_depth, self.opt.max_depth)

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
            'error_unsup_bef_update': error_unsup_bef_update,
        }
            
        return outputs, metrics, losses 