import cv2
import numpy as np
import os
import torch
import os
import copy
import time
import math
import numpy as np
import cv2
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from skimage import transform
import matplotlib.cm
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
import datasets
import datasets.kitti_c
import networks
from layers import *
from utils.utils import *
from options import MonodepthOptions
from tqdm import tqdm
import pdb
import torchvision
from utils.utils import save_tensor_as_image, calculate_batch_image_entropy, calculate_batch_edge_density
from freq_aware_depth import add_autoblured_inputs

        
def load_model(load_path, model_name, models_to_load):
    """Load model(s) from disk
    """
    load_path = os.path.expanduser(load_path)

    assert os.path.isdir(load_path), \
        "Cannot find folder {}".format(load_path)
    print("loading model from folder {}".format(load_path))

    for n in models_to_load:
        print("Loading {} weights...".format(n))
        path = os.path.join(load_path, "{}.pth".format(n))
        model_dict = model_name[n].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        if n == 'encoder':
            model_name[n].load_state_dict(model_dict, strict=False)
        else:
            model_name[n].load_state_dict(model_dict)

reg_models = {}
reg_models["encoder"] = networks.ResnetEncoder(
    50, False)
reg_models["encoder"] = torch.nn.DataParallel(reg_models["encoder"])
reg_models["encoder"].cuda()
reg_models["depth"] = networks.DepthDecoder(reg_models["encoder"].module.num_ch_enc, [0,1,2,3])
reg_models["depth"] = torch.nn.DataParallel(reg_models["depth"])
reg_models["depth"].cuda()

reg_path = './exp_logs/kitti_unsup/models/weights_19'
reg_model_folder = os.path.join(reg_path)
load_model(reg_model_folder, reg_models, ['encoder', 'depth', 'pose_encoder', 'pose'])


def match_orb_features(img1, img2):

    # Create ORB feature detector
    orb = cv2.ORB_create()
    
    # Detect and compute ORB features for each image
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    # Create BFMatcher instance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match ORB features between the two images
    matches = bf.match(des1, des2)
    
    # Get the distances between corresponding feature points
    distances = [match.distance for match in matches]
    
    return np.array(distances)


# Example usage
img1 = cv2.imread('playground/dgp/dgp_first_images/0.png')
img2 = cv2.imread('playground/dgp/dgp_first_images/1.png')

distances = match_orb_features(img1, img2)
mean_distance = np.mean(distances)

print(f"Mean distance between corresponding ORB features: {mean_distance:.2f}")

def match_orb_features(self, img1_, img2_, pix_coords):
    # Convert PyTorch tensors to numpy arrays
    img1 = img1_[0].permute(1, 2, 0).byte().numpy()
    img2 = img2_[0].permute(1, 2, 0).byte().numpy()
    


    # Create ORB feature detector
    orb = cv2.ORB_create()
    
    # Detect and compute ORB features for each image
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    flow_field = pix_coords
    
    # Modify keypoint positions in the first image based on the flow field
    # for kp in kp1:
    #     # Normalize keypoint coordinates to [-1, 1] range
    #     x = (kp.pt[0] / img1.shape[2]) * 2 - 1
    #     y = (kp.pt[1] / img1.shape[1]) * 2 - 1
        
    #     # Look up the flow field value at the keypoint location
    #     flow_x = flow_field[0, 0, int(y * (flow_field.shape[1] - 1)), int(x * (flow_field.shape[2] - 1))]
    #     flow_y = flow_field[0, 1, int(y * (flow_field.shape[1] - 1)), int(x * (flow_field.shape[2] - 1))]
        
    #     # Update the keypoint position
    #     kp.pt = (kp.pt[0] + flow_x.item() * img1.shape[2] / 2, kp.pt[1] + flow_y.item() * img1.shape[1] / 2)
    
    # Create BFMatcher instance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match ORB features between the two images
    matches = bf.match(des1, des2)
    print(matches)
    
    # Get the distances between corresponding feature points
    distances = [match.distance for match in matches]
    
    print(np.array(distances))
    
    # import sys
    # sys.exit()
    
    return np.array(distances)