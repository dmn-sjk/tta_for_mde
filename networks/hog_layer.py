import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt

def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()


class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x, ori_img=None, hog_img=None, index=0):

        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=3
        )
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=3
        )
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        phase = torch.atan2(gx_rgb, gy_rgb)
        phase = phase / self.pi * self.nbins  # [-9, 9]

        b, c, h, w = norm_rgb.shape
        out = torch.zeros(
            (b, c, self.nbins, h, w), dtype=torch.float, device=x.device
        )
        phase = phase.view(b, c, 1, h, w)
        norm_rgb = norm_rgb.view(b, c, 1, h, w)
        if self.gaussian_window:
            if h != self.gaussian_window:
                assert h % self.gaussian_window == 0, "h {} gw {}".format(
                    h, self.gaussian_window
                )
                repeat_rate = h // self.gaussian_window
                temp_gkern = self.gkern.repeat([repeat_rate, repeat_rate])
            else:
                temp_gkern = self.gkern
            norm_rgb *= temp_gkern



        out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)

        out = out.unfold(3, self.pool, self.pool)
        out = out.unfold(4, self.pool, self.pool)
        out = out.sum(dim=[-1, -2])

        out = torch.nn.functional.normalize(out, p=2, dim=2)

        return out  # B 3 nbins H//pool W//pool


    def visualize_hog_features(self, hog_tensor, name_suffix="0", mask=None):
        """
        Visualizes HOG features.

        Args:
            hog_tensor (torch.Tensor): The HOG feature tensor.
                                    Expected shape (C, nbins, H_pooled, W_pooled)
                                    or (B, C, nbins, H_pooled, W_pooled).
                                    If B is present, the first item from the batch is used.
            name_suffix (str): Suffix for the saved image filename.
            mask (torch.Tensor or np.ndarray, optional): A 2D mask with shape (H_pooled, W_pooled).
                                                        Cells where mask is True (or non-zero) will be colored red.
        """
        cell_size = self.pool
        
        if not isinstance(hog_tensor, torch.Tensor):
            raise TypeError("Input hog_tensor must be a PyTorch tensor.")
        
        # Work on a detached copy on the CPU
        work_tensor = hog_tensor.detach().clone().cpu()

        if work_tensor.ndim == 5:  # B, C, nbins, H_pooled, W_pooled
            if work_tensor.shape[0] == 0:
                print("Warning: hog_tensor batch dimension is empty. Nothing to visualize.")
                return
            work_tensor = work_tensor[0]  # Take first image in batch
        elif work_tensor.ndim != 4: # C, nbins, H_pooled, W_pooled
            raise ValueError(
                "Input hog_tensor must have 4 or 5 dimensions. "
                f"Got {work_tensor.ndim} dimensions with shape {work_tensor.shape}"
            )

        # work_tensor shape: (C, nbins, H_pooled, W_pooled)
        num_channels = work_tensor.shape[0]
        nbins = work_tensor.shape[1]
        h_pooled = work_tensor.shape[2]
        w_pooled = work_tensor.shape[3]

        if num_channels == 0 or nbins == 0 or h_pooled == 0 or w_pooled == 0:
            print(f"Warning: One of the dimensions is zero {work_tensor.shape}. Cannot visualize.")
            return

        # Average over color channels if C > 1
        if num_channels > 1:
            # Keepdims=False by default, result shape: (nbins, H_pooled, W_pooled)
            features_to_draw = work_tensor.mean(dim=0)
        else:
            # Squeeze C dim, result shape: (nbins, H_pooled, W_pooled)
            features_to_draw = work_tensor.squeeze(dim=0)

        features_to_draw = features_to_draw.numpy() # Now a NumPy array

        # Process mask
        processed_mask = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                processed_mask = mask.detach().clone().cpu().numpy()
            elif isinstance(mask, np.ndarray):
                processed_mask = mask
            else:
                raise TypeError("Mask must be a PyTorch Tensor or NumPy array.")
            
            if processed_mask.shape != (h_pooled, w_pooled):
                raise ValueError(
                    f"Mask shape {processed_mask.shape} does not match "
                    f"pooled HOG dimensions ({h_pooled}, {w_pooled})."
                )

        # Create an image canvas
        img_height = h_pooled * cell_size
        img_width = w_pooled * cell_size
        
        if img_height == 0 or img_width == 0:
            print("Resulting image dimensions are zero. Cannot visualize.")
            return

        fig, ax = plt.subplots(1, 1, figsize=(max(4, img_width / 80.0), max(4, img_height / 80.0)))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(0, img_width)
        ax.set_ylim(img_height, 0)  # Y-axis inverted (0 at top, increases downwards)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"HOG Features {name_suffix}")

        pi = math.pi

        for h_idx in range(h_pooled):
            for w_idx in range(w_pooled):
                cell_histogram = features_to_draw[:, h_idx, w_idx]  # Histogram for the current cell

                center_x = (w_idx + 0.5) * cell_size
                center_y = (h_idx + 0.5) * cell_size

                line_color = 'black' # Default color
                if processed_mask is not None and processed_mask[h_idx, w_idx]:
                    line_color = 'red'

                for bin_idx in range(nbins):
                    magnitude = cell_histogram[bin_idx]
                    
                    # Scale magnitude for line length. Using sqrt for better visibility of smaller magnitudes.
                    # Max HOG feature magnitude per bin is 1 due to L2 normalization over bins.
                    # Scaled_line_length will be at most cell_size * 0.45.
                    # The line drawn is 2 * scaled_line_length, so max length is cell_size * 0.9.
                    scaled_line_length_half = math.sqrt(max(0,magnitude)) * (cell_size * 0.45)

                    if scaled_line_length_half < 0.1:  # Skip if too small to draw effectively
                        continue

                    # Angle for this bin (unsigned, 0 to pi), measured CCW from positive Y-axis.
                    # Positive Y-axis points downwards in this plot's coordinate system.
                    angle_rad_from_y_ccw = (bin_idx + 0.5) * (pi / nbins)

                    # Components of the line segment vector (dx, dy)
                    # dx is change in x (positive right)
                    # dy is change in y (positive down, as per angle_rad_from_y_ccw definition)
                    line_dx_component = math.sin(angle_rad_from_y_ccw)
                    line_dy_component = math.cos(angle_rad_from_y_ccw)

                    start_x = center_x - line_dx_component * scaled_line_length_half
                    end_x = center_x + line_dx_component * scaled_line_length_half
                    start_y = center_y - line_dy_component * scaled_line_length_half
                    end_y = center_y + line_dy_component * scaled_line_length_half

                    ax.plot([start_x, end_x], [start_y, end_y], 
                            color=line_color, 
                            linewidth=1.5, 
                            marker='None')
        plt.savefig(f"playground_new/hog_features/waymo_all6/{name_suffix}.png", bbox_inches='tight', pad_inches=0)
        plt.close()