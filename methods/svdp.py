
from methods.base import BaseAdaptation
from methods import register_method
from utils.utils import inputs_to_device
from layers import DEPTH_METRIC_NAMES, DEPTH_METRIC_NAMES_LOCAL, DEPTH_METRIC_NAMES_UNSUP
from layers import disp_to_depth, update_ema_variables, compute_depth_errors_adadepth
from networks import get_supervised_models
from utils.losses import DepthLoss
from networks.sparse_prompter import SparsePrompter_uncertainty
from utils.encoder_decoder_merge import EncoderDecoderMerge
from utils.utils import save_tensor_as_image

import copy
import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import zoom
import torch.nn.functional as F
from utils.svdp_augs import SVDPMultiScaleFlipAug, Resize, RandomFlip
import os


@register_method(name='svdp')
class SVDP(BaseAdaptation):
    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        models = get_supervised_models(self.opt)
        model = self.configure_model(models)
        self.model = SVDPModel(model, 
                                # TODO: adjust the size, should be the same as the original image size 
                                img_height=self.opt.height, img_width=self.opt.width,
                                patch_size=self.opt.patch_size
                               )
        
        if self.opt.sup_model == 'dpt':
            # NOTE/TODO: the deepcopy does not work properly on DPT because of the hooks which assign the value of activations to the global variable,
            # which is then referenced in pretrained.activations. After deepcopy, pretrained.activations change the address and when the global variable is updated,
            # the new pretrained.activations does not match the global variable.
            # This way, still the same global variable is referenced, which might also cause problems with some parallelism.
            ema_models = get_supervised_models(self.opt)
            model = self.configure_model(ema_models)
            self.ema_model = SVDPModel(model, 
                                    # TODO: adjust the size, should be the same as the original image size 
                                    img_height=self.opt.height, img_width=self.opt.width,
                                    patch_size=self.opt.patch_size
                                )
            anchor_models = get_supervised_models(self.opt)
            model = self.configure_model(anchor_models)
            self.anchor_model = SVDPModel(model, 
                                    # TODO: adjust the size, should be the same as the original image size 
                                    img_height=self.opt.height, img_width=self.opt.width,
                                    patch_size=self.opt.patch_size
                                )
        else:
            self.ema_model = copy.deepcopy(self.model)
            self.anchor_model = copy.deepcopy(self.model)

        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.detach_()
        mp = list(self.model.parameters())
        mcp = list(self.ema_model.parameters())
        n = len(mp)
        for i in range(0, n):
            mcp[i].data[:] = mp[i].data[:].clone()

        self.anchor_model.eval()
        self.anchor = copy.deepcopy(self.model.state_dict())

        self.loss = DepthLoss()
       
       # NOTE:SVDP executes the below on each domain again
        param_model_list = []
        param_prompt_list = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and "prompt" in name:
                param_prompt_list.append(param)
            elif param.requires_grad and "prompt" not in name:
                param_model_list.append(param)
            else:
                param.requires_grad=False
        
        # two sets of default SVDP: 
        # 1st:
        # self.prompt_lr = 0.0001
        # self.model_lr = 0.0001
        self.prompt_lr = self.opt.learning_rate
        self.model_lr = self.opt.learning_rate
        self.ema_rate = 0.999
        self.prompt_sparse_rate=0.001
        # self.scale = 0.1
        # 2nd:
        # self.prompt_lr = 0.01
        # self.model_lr = 0.0003
        # self.ema_rate = 0.999
        # self.prompt_sparse_rate=0.0001
        # self.scale = 0.01
        
        # my after experiments:
        if self.opt.sup_model == 'dpt':
            self.scale = 1e-5
        else:
            self.scale = 1e-3

        self.optimizer = torch.optim.Adam([{"params": param_prompt_list, "lr": self.prompt_lr},
                                    {"params": param_model_list, "lr": self.model_lr}],
                                    lr=1e-5, betas=(0.9, 0.999))
        

    def process_batch(self, inputs):
        inputs_to_device(inputs, self.opt.device) 
                    # model,
                    # data_loader,
                    # show=False,
                    # out_dir=None,
                    # efficient_test=False,
                    # anchor=None,
                    # ema_model=None,
                    # anchor_model=None,
                    # dynamic_ema=False,
                    # dynamic_prompt_only=False,

        # for i, data in enumerate(data_loader):
        self.model.eval()
        self.ema_model.eval()
        assert isinstance(self.get_model_dropout_layer(self.ema_model), nn.Dropout)
        assert isinstance(self.model.prompt, SparsePrompter_uncertainty)
        self.get_model_dropout_layer(self.ema_model).train()
        self.model.prompt.if_mask = False

        with torch.no_grad():
            # data_one = {
            #     'img_metas': [data['img_metas'][4] for i in range(1)],
            #     'img': [data['img'][4] for i in range(1)]
            # }
            # _, _, _, unc_all = ema_model(return_loss=False, svdp = True, dropout_num=10, **data_one)
            # NOTE: remember about svdp = True and dropout_num=10
            # TODO: calculate uncertainty for depth using dropout
            _, variance = self.ema_model((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std, svdp=True)

            # _, prob_anchor= anchor_model(return_loss=False, **data_one)
            # prob_anchor = self.anchor_model((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std)
            # NOTE: I guess just do not use it for depth
            # mask = (prob_anchor[0] > 0.69).astype(np.int64) # 0.74 was the 5% quantile for cityscapes, therefore we use 0.69 here

            # fake uncertainty metric for depth
            uncertainty = variance.squeeze().cpu().numpy()

            # Domain Prompt Placement
            # model.module.backbone.prompt.if_mask = True
            # model.module.backbone.prompt.update_uncmap(uncertainty[0])
            # model.module.backbone.prompt.update_mask()
            self.model.prompt.if_mask = True
            self.model.prompt.update_uncmap(uncertainty)
            self.model.prompt.update_mask()

            # ema_model.module.backbone.prompt.if_mask = True
            # ema_model.module.backbone.prompt.update_uncmap(uncertainty[0])
            # ema_model.module.backbone.prompt.update_mask()
            # ema_model.eval()
            self.ema_model.prompt.if_mask = True
            self.ema_model.prompt.update_uncmap(uncertainty)
            self.ema_model.prompt.update_mask()
            self.ema_model.eval()

            # result, probs, preds = ema_model(return_loss=False,  **data)
            result, _ = self.ema_model((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std, forward_augs=True)
            # NOTE: don't do it since we don't have mask for depth (see above)
            # result = [(mask*preds[4][0] + (1.-mask)*result[0]).astype(np.int64)]
            weight = 1.
            

        # if isinstance(result, list):
        #     if len(data['img']) == 14:
        #         img_id = 4 #The default size without flip 
        #     else:
        #         img_id = 0

        #     loss = model.forward(return_loss=True, img=data['img'][img_id], img_metas=data['img_metas'][img_id].data[0], gt_semantic_seg=torch.from_numpy(result[0]).cuda().unsqueeze(0).unsqueeze(0))
        #     results.extend(result)
        # else:
        #     results.append(result)

        depth_pred = self.model((inputs["color_uncrop", 0, 0]-self.opt.mean)/self.opt.std, forward_train=True)
        loss = self.loss(depth_pred, result)
        loss = torch.mean(loss * weight)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Domain Prompt Updating
        # NOTE: the higher the uncertainty, the more the prompt is updated (the lower prompt rate is, the less the ema teacher matters)
        prompt_rate = self.ema_rate - np.average(uncertainty) * self.scale
        update_ema_variables(model = self.model, ema_model = self.ema_model, alpha=self.ema_rate, 
                            alpha_prompt = prompt_rate)
        for nm, m in self.model.named_modules():
            for npp, p in m.named_parameters():
                if npp in ['weight', 'bias'] and p.requires_grad:
                    mask = (torch.rand(p.shape)<0.01).float().cuda()
                    with torch.no_grad():
                        p.data = self.anchor[f"{nm}.{npp}"] * mask + p * (1.-mask)
        
        error = list(compute_depth_errors_adadepth(self.opt, inputs['depth_gt_uncrop'], result, median_scaling=True))

        for idx, term in enumerate(error):
            error[idx] = term.detach().cpu().numpy()

        outputs = {}
        outputs['depth'] = result

        losses = {}
        losses['loss'] = loss.detach().cpu()

        metrics = {
            'error': error,
        }

        return outputs, metrics, losses
    
    def configure_model(self, models):
        model = EncoderDecoderMerge(models["encoder"], models["depth"])
        if self.opt.sup_model == 'dpt':
            # Add dropout layer to model head at index 3
            head_layers = list(model.decoder.module.scratch.output_conv)
            head_layers.insert(3, nn.Dropout(0.5))
            head_layers[3].eval()
            model.decoder.module.scratch.output_conv = nn.Sequential(*head_layers)
        elif self.opt.sup_model == 'newcrf':
            model.decoder.module.disp_head1.conv1 = nn.Sequential(nn.Dropout(0.5), model.decoder.module.disp_head1.conv1)
            model.decoder.module.disp_head1.conv1[0].eval()
        else:
            raise NotImplementedError("Invalid model type")
        return model

    def get_model_dropout_layer(self, model):
        if self.opt.sup_model == 'dpt':
            return model.model.decoder.module.scratch.output_conv[3]
        elif self.opt.sup_model == 'newcrf':
            return model.model.decoder.module.disp_head1.conv1[0]
        else:
            raise NotImplementedError("Invalid model type")

class SVDPModel(nn.Module):
    def __init__(self, model, prompt_sparse_rate=0.25, img_height=1080, img_width=1920, align_corners=False, patch_size=32):
        super(SVDPModel, self).__init__()
        self.model = model
        # TODO: img_scale should probaly be half of the orginal size
        self.svdp_augs = SVDPMultiScaleFlipAug(
            img_scale=(img_height, img_width), # TODO: adjust the size, should be the same as the original image size 
            img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
            flip=False,
            patch_size=patch_size,
            transforms=[
                Resize(keep_ratio=True),
                RandomFlip(),
                # dict(type='Normalize', **img_norm_cfg), # assume the images are already normalized form dataloader
            ])
        scales = self.svdp_augs.img_scale
        if (img_height, img_width) not in scales:
            scales.append((img_height, img_width))
        # NOTE: important when using flip (second duplicate of scale is for flipped mask)
        scales = [scale for scale in scales for _ in range(2)]
        self.prompt = SparsePrompter_uncertainty(sparse_rate=prompt_sparse_rate, 
                                                 scales=scales,
                                                 shape=(img_height, img_width))

        self.ori_shape = (img_height, img_width)
        self.align_corners = align_corners

    def forward(self, x, svdp=False, position=None, forward_train=False, forward_augs=False):
        if svdp:
            # TODO: make sure that images have the same shapes
            if forward_augs:
                x = self.aug_test_svdp(x)
            else:
                x = self.simple_test_svdp(x)
        else:
            if forward_train and forward_augs:
                raise ValueError("forward_train and forward_augs cannot be both True")

            if forward_train:
                position = (0, x.shape[-2], 0, x.shape[-1])
                x = self.prompt(x, position=position)

                x = self.model(x)
                x = F.interpolate(x, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners) 
            elif forward_augs:
                x = self.aug_test(x)
            else:
                x = self.simple_test(x)
        return x
    
    def simple_test(self, x, rescale=True):
        """Simple test with single image."""
        output = self.inference(x, rescale)
        return output
    
    def aug_test(self, x, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale

        augmented_x = self.svdp_augs({'img': x})
       
        sum_pred = None 
        outs = []
        for i in range(len(augmented_x['img'])):
            cur_out = self.inference(augmented_x['img'][i], 
                                           rescale, 
                                           flip=augmented_x['flip'][i], 
                                           flip_direction=augmented_x['flip_direction'][i])
            if sum_pred is None:
                sum_pred = cur_out.clone()
            else:
                sum_pred += cur_out
                
            outs.append(cur_out.cpu())
        output = sum_pred / len(augmented_x['img'])
        outs = torch.stack(outs)
        variance = torch.var(outs, dim=0)
        # uncertainty = np.sum(variance, axis=1)
        return output, variance
    
    def simple_test_svdp(self, x, dropout_num=10):
        """Simple test with single image.
        Args:
            dropout_num (int): The number of dropout samples.
        """

        output_lst = self.inference_svdp(x, dropout_num)
        outs = torch.stack(output_lst)
        variance = torch.var(outs, dim=0)
        # uncertainty = np.sum(variance, axis=1)
        return outs, variance
    
    def inference_svdp(self, x, dropout_num=10):
        position = (0, x.shape[-2], 0, x.shape[-1])
        x = self.prompt(x, position=position)

        preds_lst = self.encode_decode_svdp(x, dropout_num)
        return preds_lst

    def encode_decode_svdp(self, x, drop_num):
        """Encode images with backbone and decode into a depth map of the same size as input."""
        # TODO: can divide model into encoder and decoder and forward through encoder only once before loop (dropout in head)
        All_output = []
        for idx in range(drop_num):
            out = self.model(x)
            out = F.interpolate(out, size=self.ori_shape, mode='bilinear', align_corners=self.align_corners)
            
            All_output.append(out)
        return All_output

    def aug_test_svdp(self, x, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        
        augmented_x = self.svdp_augs({'img': x})

        sum_pred = None
        outs = []
        for i in range(len(augmented_x['img'])):
            cur_out = self.inference(augmented_x['img'][i], 
                                    rescale, 
                                    flip=augmented_x['flip'][i], 
                                    flip_direction=augmented_x['flip_direction'][i])
            if sum_pred is None:
                sum_pred = cur_out.clone()
            else:
                sum_pred += cur_out
            
            outs.append(cur_out.cpu())
        output = sum_pred / len(augmented_x['img'])
        outs = torch.stack(outs)
        variance = torch.var(outs, axis=0)
        # uncertainty = np.sum(variance, axis=1)
        return output, outs, variance
    
    def inference(self, x, rescale=False, flip=False, flip_direction=None):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        position = (0, x.shape[-2], 0, x.shape[-1])
        if flip and flip_direction != 'horizontal':
            raise NotImplementedError("Only horizontal flip supported in prompt")
        x = self.prompt(x, flip=flip, position=position)

        output = self.encode_decode(x)
        if rescale:
            # TODO: check shape
            output = F.interpolate(output, size=self.ori_shape, mode='bilinear', align_corners=self.align_corners) 
        
        if flip:
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = torch.flip(output, [3])
            elif flip_direction == 'vertical':
                output = torch.flip(output, [2])

        return output
    
    def encode_decode(self, x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        # NOTE: they give position here from slide inference, I used whole inference (refer to EncoderDecoder.slide_inference())
        # I believe that what they pass here is only the slices of the images on which they infer at the time (the edges of image on which 
        # to prompts can be put, in my case (whole inference without slices) I can just pass img size). - Did it in self.forward().
        # x = self.extract_feat(x) # position)
        # out = self.decode_head(x)
        out = self.model(x)
        # TODO: check shape
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=self.align_corners) 
        return out
