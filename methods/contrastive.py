
from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
import torch
from layers import disp_to_depth, compute_depth_errors_adadepth, compute_depth_errors
from networks import get_supervised_models, get_self_supervised_models
from utils.svdp_augs import SVDPMultiScaleFlipAug, Resize, RandomFlip
from utils.losses import TripletLoss
from layers import update_ema_variables

import torch.nn.functional as F
import copy


@register_method(name='contrastive')
class Contrastive(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.depth_features = None
        def hook_fn(name):
            def hook(module, input, output):
                self.depth_features = output
            return hook

        if self.opt.model_type == "supervised":
            self.models = get_supervised_models(self.opt)
            rel_depth = False
        
            if self.opt.sup_model == 'newcrf':
                self.models['depth'].module.crf0.register_forward_hook(hook_fn('disp_head1'))
            else:
                raise Exception("Model not supported")
        
        else:
            self.models = get_self_supervised_models(self.opt)
            rel_depth = True
            self.models['depth'].module.convs.upconv01.register_forward_hook(hook_fn('disp_head1'))

        # Print all possible nodes that can be hooked in self.models['depth']
        # def print_model_nodes(model, prefix=''):
        #     for name, module in model.named_children():
        #         full_name = f"{prefix}.{name}" if prefix else name
        #         print(f"Node: {full_name}")
        #         print_model_nodes(module, full_name)
        
        # print("=== Possible nodes for hooks in self.models['depth'] ===")
        # print_model_nodes(self.models['depth'])
        # print("=== End of nodes list ===")
        
       

        # a = torch.randn(1, 3, self.opt.height, self.opt.width)
        # b = self.models['encoder'](a)
        # c = self.models['depth'](b)
        # print(c.shape)
        # print(len(self.depth_features))
        # print(self.depth_features[0].shape)
        # print(self.depth_features.shape)
        # print(self.depth_features.max()V)
        # raise Exception("Stop here")


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
 
        self.loss = TripletLoss(margin=1.0, reduction='mean')
        self.miner = TripletMiner(pos_thr=self.opt.pos_thr, neg_thr=self.opt.neg_thr, rel_depth=rel_depth)

    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device)
        
        input_img = (inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std
        input_batch = copy.deepcopy(input_img)
        for frame_id in self.opt.frame_ids[1:]:
            input_batch = torch.cat([input_batch, (inputs["color_uncrop", frame_id, 0]-self.opt.mean)/self.opt.std], dim=0)

        features = self.models["encoder"](input_batch)
        pred_depth = self.models["depth"](features)
        if self.opt.model_type == "self-supervised":
            _, pred_depth = disp_to_depth(pred_depth[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
 
        pred_depth_mining = F.interpolate(pred_depth, self.depth_features.shape[-2:], mode="bilinear", align_corners=False)
        anchors, positives, negatives = self.miner.mine(self.depth_features, pred_depth_mining)
        loss = self.loss(anchors, positives, negatives).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for model, model_ema in zip(self.models.values(), self.models_ema.values()):
        #     update_ema_variables(model, model_ema, 0.999)

        with torch.no_grad():
            features = self.models["encoder"](input_img)
            depth_out = self.models["depth"](features)
            if self.opt.model_type == "self-supervised":
                _, depth_out = disp_to_depth(depth_out[("disp", 0)], self.opt.min_depth, self.opt.max_depth)
            

        error_local = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=True))
        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], depth_out, median_scaling=False))

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()
        for idx, term in enumerate(error_local):
            error_local[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = depth_out

        losses = {}
        
        metrics = {
            'error': error,
            'error_local': error_local,
        }
        
        return outputs, metrics, losses

class TripletMiner:
    def __init__(self, pos_thr, neg_thr, rel_depth=False):
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        self.rel_depth = rel_depth

    def mine(self, depth_features, pred_depth):
        # Get the shape of the tensors
        _, C, H, W = depth_features.shape
        
        assert pred_depth.shape[-2:] == depth_features.shape[-2:], f"pred_depth.shape[-2:] = {pred_depth.shape[-2:]} != depth_features.shape[-2:] = {depth_features.shape[-2:]}"

        pred_depth = pred_depth.squeeze(1)

        # Choose 1000 random pixel coordinates
        num_samples = 100
        
        # flatten image and batch dimension
        flat_depth_features = depth_features.view(depth_features.shape[1], -1)
        flat_pred_depth = pred_depth.view(-1)
        
        anchor_indices = torch.randint(0, flat_depth_features.shape[-1], (num_samples,), device=depth_features.device)
 
        # Get the depth values at the anchor points
        anchor_depth_values = flat_pred_depth[anchor_indices]  # Shape: (num_samples)
        
        # Compute pairwise differences between all anchor depth values
        # Reshape for broadcasting: (num_samples, 1) - (1, num_samples) = (num_samples, num_samples)
        anchor_depths_i = anchor_depth_values.view(-1, 1)
        depths_j = flat_pred_depth.view(1, -1)
        pairwise_depth_diff = torch.abs(anchor_depths_i - depths_j)  # Shape: (B, num_samples, num_samples)
        
        num_of_triplets_per_anchor = 100
        
        # Get the positive and negative indices
        if self.rel_depth:
            pairwise_depth_diff = pairwise_depth_diff / (flat_pred_depth.max() - flat_pred_depth.min())
        
        positives_mask = pairwise_depth_diff < self.pos_thr
        negatives_mask = pairwise_depth_diff > self.neg_thr
        
        
        # Reshape masks for easier processing
        # Shape: (batch_size, num_samples, H*W)
        positives_mask = positives_mask.to(torch.bool)
        negatives_mask = negatives_mask.to(torch.bool)
        
        # Initialize tensors to store the final triplets
        anchors = []
        positives = []
        negatives = []

        anchor_features = flat_depth_features[:, anchor_indices]
        
        for i in range(num_samples):
            # Get positive and negative indices for this anchor
            pos_indices = torch.nonzero(positives_mask[i]).squeeze(-1)
            neg_indices = torch.nonzero(negatives_mask[i]).squeeze(-1)

            # Skip if we don't have any positives or negatives
            if pos_indices.numel() == 0 or neg_indices.numel() == 0:
                continue
            
            # Randomly sample indices if we have more than needed
            if pos_indices.numel() > num_of_triplets_per_anchor:
                pos_indices = pos_indices[torch.randperm(pos_indices.numel(), device=pos_indices.device)[:num_of_triplets_per_anchor]]
            
            if neg_indices.numel() > num_of_triplets_per_anchor:
                neg_indices = neg_indices[torch.randperm(neg_indices.numel(), device=neg_indices.device)[:num_of_triplets_per_anchor]]
            
            # Get the minimum number of triplets we can form
            num_triplets = min(pos_indices.numel(), neg_indices.numel())
            
            # print(num_triplets)
            
            # Use only as many as we need
            pos_indices = pos_indices[:num_triplets]
            neg_indices = neg_indices[:num_triplets]
            
            # Get the features
            anchor_feat = anchor_features[:, i].unsqueeze(0).expand(num_triplets, -1)
            pos_feat = flat_depth_features[:, pos_indices].t()
            neg_feat = flat_depth_features[:, neg_indices].t()
            
            # Add to our lists
            anchors.append(anchor_feat)
            positives.append(pos_feat)
            negatives.append(neg_feat)
        
        # Concatenate all triplets
        if anchors:
            anchors = torch.cat(anchors, dim=0)
            positives = torch.cat(positives, dim=0)
            negatives = torch.cat(negatives, dim=0)
        else:
            # Handle the case where no valid triplets were found
            anchors = torch.zeros((0, C), device=depth_features.device)
            positives = torch.zeros((0, C), device=depth_features.device)
            negatives = torch.zeros((0, C), device=depth_features.device)

        return anchors, positives, negatives
