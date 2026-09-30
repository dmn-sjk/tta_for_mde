from .resnet_encoder import ResnetEncoder, skip_resnet
from .depth_decoder import DepthDecoder
from .pose_decoder import PoseDecoder
from .pose_cnn import PoseCNN
from .NewCRFDepth import trans_backbone, NewCRFDepth
from .dpt import DPTDepthModel
from .unet import get_unet
import torch
import os
import torchvision
from .depth_anything_v2 import get_depth_anything_v2_metric


class DummyEncoder(torch.nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()

    def forward(self, x):
        return x


def get_supervised_models(opt):
    models = {}
    
    if opt.sup_model == 'newcrf':
        models["encoder"] = trans_backbone(
            'large07', './models/swin_large_patch4_window7_224_22k.pth')
        models["encoder"] = torch.nn.DataParallel(models["encoder"])
        models["depth"] = NewCRFDepth(version = 'large07', max_depth = opt.MAX_DEPTH)
        models["encoder"].to(opt.device)
        models["depth"] = torch.nn.DataParallel(models["depth"])
        models["depth"].to(opt.device)
        load_model_list = ['encoder', 'depth']

        if opt.sup_model_path is not None:
            load_model(opt.sup_model_path, models, load_model_list)
        opt.patch_size = 32

    elif opt.sup_model == 'dpt':
        # TODO: make it better, without DummyEncoder
        # TODO: make sure that std and mean are correct for this model
        net_w = 1216
        net_h = 352
        model = DPTDepthModel(
            path=opt.sup_model_path,
            scale=0.00006016,
            shift=0.00579,
            invert=True, # invert means it will return depth, not disparity
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        models["encoder"] = DummyEncoder()
        models["encoder"] = torch.nn.DataParallel(models["encoder"])
        models["encoder"].to(opt.device)
        models["depth"] = model
        models["depth"] = torch.nn.DataParallel(models["depth"])
        models["depth"].to(opt.device)
        opt.patch_size = 16
    elif opt.sup_model == 'unet':
        # TODO: handle encoder and depth head separately
        model = get_unet(opt.sup_model_path)
        models["encoder"] = DummyEncoder()
        models["encoder"] = torch.nn.DataParallel(models["encoder"])
        models["encoder"].to(opt.device)
        models["depth"] = model
        models["depth"] = torch.nn.DataParallel(models["depth"])
        models["depth"].to(opt.device)
    elif 'depth_anything_v2' in opt.sup_model:
        # TODO: handle encoder and depth head separately
        model = get_depth_anything_v2_metric(ckpt_dir=opt.sup_model_path,
                                             encoder=opt.sup_model.split('_')[-1])
        
        models["encoder"] = DummyEncoder()
        models["encoder"] = torch.nn.DataParallel(models["encoder"])
        models["encoder"].to(opt.device)
        models["depth"] = model
        models["depth"] = torch.nn.DataParallel(models["depth"])
        models["depth"].to(opt.device)
        opt.patch_size = 14
    else:
        raise NotImplementedError

    return models

    
def get_self_supervised_models(opt):
    models = {}
    models["pose_encoder"] = ResnetEncoder(
        opt.num_layers,
        opt.weights_init == "pretrained",
        num_input_images=opt.num_pose_frames)
    models["pose_encoder"] = torch.nn.DataParallel(models["pose_encoder"])
    models["pose_encoder"].to(opt.device)
    models["pose"] = PoseDecoder(
        models["pose_encoder"].module.num_ch_enc,
        num_input_features=1,
        num_frames_to_predict_for=2)
    models["pose"] = torch.nn.DataParallel(models["pose"])
    models["pose"].to(opt.device)
    
    dropout = 0.0
    if opt.skip_layers:
        if opt.dropout:
            raise NotImplementedError("Skip layers with dropout not implemented")
        # overwrite _resnet function for it to output my version of ResNet with layer skips
        torchvision.models.resnet._resnet = skip_resnet 
    elif opt.dropout:
        dropout = 0.4
    
    models["encoder"] = ResnetEncoder(
        opt.num_layers, False, dropout=dropout)
    models["encoder"] = torch.nn.DataParallel(models["encoder"])
    models["encoder"].to(opt.device)
    models["depth"] = DepthDecoder(models["encoder"].module.num_ch_enc, opt.scales)
    models["depth"] = torch.nn.DataParallel(models["depth"])
    models["depth"].to(opt.device)

    reg_model_folder = os.path.join(opt.ssl_model_path)
    load_model(reg_model_folder, models, ['encoder', 'depth', 'pose_encoder', 'pose'])
    
    # opt.patch_size = None

    return models


def load_model(load_path, models, model_names_to_load):
    """Load model(s) from disk
    """
    load_path = os.path.expanduser(load_path)

    assert os.path.isdir(load_path), \
        "Cannot find folder {}".format(load_path)
    print("loading model from folder {}".format(load_path))

    for n in model_names_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_path, "{}.pth".format(n))
        model_dict = models[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        if n == 'encoder':
            models[n].load_state_dict(model_dict, strict=True)
        else:
            models[n].load_state_dict(model_dict)
