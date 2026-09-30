# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
import random


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))

def save_tensor_as_image(tensor, filename):
    # Ensure the tensor is on CPU
    tensor = tensor.cpu()
    # handle nan values
    mask = ~torch.isnan(tensor)
    tensor = tensor / torch.max(tensor[mask])
    
    # Remove the batch dimension and permute the tensor to (H, W, C)
    tensor = tensor.squeeze(0)
    
    # Scale the tensor values to the range [0, 255]
    tensor = (tensor * 255).clamp(0, 255).to(torch.uint8)
    
    # Convert the tensor to a PIL Image
    image = transforms.ToPILImage()(tensor)
    
    # Save the image
    image.save(filename)
    

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def calculate_batch_image_entropy(image_batch):
    """
    Calculate the average entropy of a batch of images stored as a PyTorch tensor.
    
    Args:
    image_batch (torch.Tensor): Input image batch tensor with shape (B, C, H, W)
    
    Returns:
    float: Average entropy of the image batch
    """
    
    # Ensure the input is a PyTorch tensor
    if not isinstance(image_batch, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    # Ensure the input is a 4D tensor (B, C, H, W)
    if image_batch.dim() != 4:
        raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")
    
    # Normalize the image tensor to [0, 255] and convert to integers
    normalized = (image_batch * 255).byte()
    
    # Reshape the tensor to (B, -1) to handle all channels and pixels for each image
    batch_size = normalized.size(0)
    flattened = normalized.view(batch_size, -1)
    
    # Initialize a list to store entropies
    entropies = []
    
    for i in range(batch_size):
        # Calculate the histogram for each image
        histogram = torch.histc(flattened[i].float(), bins=256, min=0, max=255)
        
        # Calculate the probability distribution
        prob_dist = histogram / histogram.sum()
        
        # Remove zero probabilities to avoid log(0)
        prob_dist = prob_dist[prob_dist > 0]
        
        # Calculate entropy
        entropy = -torch.sum(prob_dist * torch.log2(prob_dist))
        entropies.append(entropy.item())
    
    # Calculate and return the average entropy
    avg_entropy = sum(entropies) / batch_size
    return avg_entropy

def calculate_batch_edge_density(image_batch, threshold=100):
    """
    Calculate the average edge density of a batch of images stored as a PyTorch tensor.
    
    Args:
    image_batch (torch.Tensor): Input image batch tensor with shape (B, C, H, W)
    threshold (int): Threshold for edge detection (0-255). Default is 100.
    
    Returns:
    float: Average edge density of the image batch
    """
    
    # Ensure the input is a PyTorch tensor
    if not isinstance(image_batch, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")
    
    # Ensure the input is a 4D tensor (B, C, H, W)
    if image_batch.dim() != 4:
        raise ValueError("Input tensor must have 4 dimensions (B, C, H, W)")
    
    # Convert to grayscale if the input is RGB
    if image_batch.size(1) == 3:
        grayscale = 0.2989 * image_batch[:, 0] + 0.5870 * image_batch[:, 1] + 0.1140 * image_batch[:, 2]
        grayscale = grayscale.unsqueeze(1)
    else:
        grayscale = image_batch
    
    # Normalize the image tensor to [0, 255]
    normalized = (grayscale * 255).byte().float()
    
    # Define Sobel filters
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Move Sobel filters to the same device as the input tensor
    sobel_x = sobel_x.to(normalized.device)
    sobel_y = sobel_y.to(normalized.device)
    
    # Apply Sobel filters
    edges_x = F.conv2d(normalized, sobel_x, padding=1)
    edges_y = F.conv2d(normalized, sobel_y, padding=1)
    
    # Calculate edge magnitude
    edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)
    
    # Apply threshold
    edges = (edge_magnitude > threshold).float()
    
    # Calculate edge density for each image in the batch
    edge_densities = edges.mean(dim=[1, 2, 3])
    
    # Calculate and return the average edge density
    avg_edge_density = edge_densities.mean().item()
    return avg_edge_density

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def save_data_dict(data, path):
    
    to_save = {}
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            to_save[key] = np.array(val.item())
        else:
            to_save[key] = np.array(val)
            
    np.save(path, to_save)

def inputs_to_device(inputs, device):
    for key, ipt in inputs.items():
        if isinstance(ipt, torch.Tensor):
            inputs[key] = ipt.to(device)
    
def plot_trajectory(transforms):
    # Extract positions and orientations
    x = [T[0, 3] for T in transforms]
    y = [T[1, 3] for T in transforms]
    dx_x = [T[0, 0] for T in transforms]  # X-axis (forward) direction components
    dy_x = [T[1, 0] for T in transforms]
    dx_y = [T[0, 1] for T in transforms]  # Y-axis (left) direction components
    dy_y = [T[1, 1] for T in transforms]

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot trajectory with order numbers
    plt.plot(x, y, 'b-', label='Trajectory')
    for i, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi + 0.1, yi + 0.1, str(i), fontsize=8, color='black',  # Offset text slightly
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Add orientation arrows
    q_x = plt.quiver(x, y, dx_x, dy_x, color='r', scale=1, 
                    width=0.003, angles='xy', scale_units='inches', label='X (Forward)')
    q_y = plt.quiver(x, y, dx_y, dy_y, color='g', scale=1,
                    width=0.003, angles='xy', scale_units='inches', label='Y (Left)')

    # Configure plot
    plt.title('Vehicle Trajectory with Orientation Arrows and Point Order')
    plt.xlabel('Global X')
    plt.ylabel('Global Y')
    plt.axis('equal')
    plt.grid(True)
    plt.legend(handles=[q_x, q_y, plt.Line2D([], [], color='b', label='Trajectory')])
    plt.savefig('test_corr.png')   

# Create a colorbar reference image for the depth maps
def create_colorbar(vmin=0, vmax=80, cmap=matplotlib.cm.plasma, title="Depth (m)", log_scale=False):
    # Create figure with more height to accommodate title and labels
    fig = plt.figure(figsize=(8, 2.0))
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.4])  # [left, bottom, width, height]
    
    if log_scale:
        # For logarithmic scale
        epsilon = 1e-6
        log_vmin = np.sign(vmin) * np.log1p(np.abs(vmin) + epsilon)
        log_vmax = np.sign(vmax) * np.log1p(np.abs(vmax) + epsilon)
        
        # Create tick positions in log space
        if vmin < 0 and vmax > 0:
            # Handle case with both positive and negative values
            # Use fewer ticks to avoid overlap
            neg_ticks = np.array([-40, -20, -5])  # Custom negative ticks
            pos_ticks = np.array([5, 20, 40])     # Custom positive ticks
            ticks = np.concatenate([neg_ticks, [0], pos_ticks])
        else:
            # All positive or all negative - use more ticks
            ticks = np.sign(vmin) * np.logspace(
                np.log10(np.abs(vmin) + epsilon), 
                np.log10(np.abs(vmax) + epsilon), 
                5
            )
        
        # Create normalized values for the colorbar
        norm = matplotlib.colors.Normalize(vmin=log_vmin, vmax=log_vmax)
        
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        
        # Convert ticks from original scale to log scale
        log_ticks = np.sign(ticks) * np.log1p(np.abs(ticks) + epsilon)
        cb.set_ticks(log_ticks)
        cb.ax.set_xticklabels([f'{t:.1f}' for t in ticks])
    else:
        # Linear scale (original code)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
        cb.set_ticks(np.linspace(vmin, vmax, 5))
    
    # Make title and labels larger and position title higher
    cb.set_label(title, fontsize=14, labelpad=10)
    cb.ax.tick_params(labelsize=12)
    
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))