import torch
import os

from .dpt import DepthAnythingV2


def get_depth_anything_v2_metric(ckpt_dir='ckpts', encoder='vits', dataset='vkitti'):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    # dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    if dataset == 'hypersim':
        max_depth = 20 # 20 for indoor model, 80 for outdoor model
    else:
        max_depth = 80

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, f'depth_anything_v2_metric_{dataset}_{encoder}.pth'), map_location='cpu'))
    model.eval()
    return model
