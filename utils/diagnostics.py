from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def compute_depth_out_spectral_diagnostics(depth_out):
    # Run the diagnostic on detached float32 CPU tensors to avoid cuFFT
    # failures on model-output CUDA tensors while keeping training unchanged.
    depth_out_diag = depth_out.detach().to(dtype=torch.float32, device="cpu")
    centered_depth = depth_out_diag - depth_out_diag.mean(dim=(-2, -1), keepdim=True)

    fft_depth = torch.fft.fftshift(torch.fft.fft2(centered_depth, norm="ortho"), dim=(-2, -1))
    fft_magnitude = fft_depth.abs()

    _, _, height, width = depth_out_diag.shape
    yy = torch.linspace(-1.0, 1.0, height, device=depth_out_diag.device)
    xx = torch.linspace(-1.0, 1.0, width, device=depth_out_diag.device)
    radial_freq = torch.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    high_freq_mask = (radial_freq >= 0.5).view(1, 1, height, width)
    non_dc_mask = (radial_freq > 0.0).view(1, 1, height, width)

    high_freq_energy = fft_magnitude[high_freq_mask.expand_as(fft_magnitude)].mean()
    non_dc_energy = fft_magnitude[non_dc_mask.expand_as(fft_magnitude)].mean()
    high_freq_ratio = high_freq_energy / (non_dc_energy + 1e-8)

    laplacian_kernel = depth_out_diag.new_tensor([
        [0.0, 1.0, 0.0],
        [1.0, -4.0, 1.0],
        [0.0, 1.0, 0.0],
    ]).view(1, 1, 3, 3)
    laplacian_response = F.conv2d(centered_depth, laplacian_kernel, padding=1)

    return {
        # "diag/depth_out_fft_high_freq_energy": high_freq_energy.detach().cpu().item(),
        "diag/depth_out_fft_high_freq_ratio": high_freq_ratio.detach().cpu().item(),
        # "diag/depth_out_laplacian_abs_mean": laplacian_response.abs().mean().detach().cpu().item(),
    }


def _as_depth_batch(depth_map):
    """
    Convert common depth map layouts to (B, 1, H, W) float tensors.
    """
    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().to(dtype=torch.float32)
    else:
        depth = torch.as_tensor(depth_map, dtype=torch.float32)

    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
    elif depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth.permute(2, 0, 1).unsqueeze(0)
        elif depth.shape[0] == 1:
            depth = depth.unsqueeze(0)
        else:
            depth = depth.unsqueeze(1)
    elif depth.ndim == 4:
        if depth.shape[1] != 1:
            raise ValueError(f"Expected depth map with one channel, got shape {tuple(depth.shape)}")
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D depth map, got shape {tuple(depth.shape)}")

    return depth


def _depth_forward_gradients(depth_map):
    grad_x = depth_map[..., :, 1:] - depth_map[..., :, :-1]
    grad_y = depth_map[..., 1:, :] - depth_map[..., :-1, :]
    return grad_x, grad_y


def compute_total_variation(depth_map, isotropic=False, reduction="mean"):
    """
    Compute raw/blind total variation for a depth map.

    Args:
        depth_map (torch.Tensor or np.ndarray): Depth map with shape (H, W),
            (1, H, W), (B, H, W), or (B, 1, H, W).
        isotropic (bool): If True, use sqrt(dx^2 + dy^2). Otherwise use
            anisotropic TV: |dx| + |dy|.
        reduction (str): "mean" averages over the batch, "sum" returns the
            summed TV over all batch elements.

    Returns:
        float: Total variation score.
    """
    depth = _as_depth_batch(depth_map)
    grad_x, grad_y = _depth_forward_gradients(depth)

    if isotropic:
        grad_x_common = grad_x[..., :-1, :]
        grad_y_common = grad_y[..., :, :-1]
        tv_per_sample = torch.sqrt(grad_x_common.square() + grad_y_common.square() + 1e-12).sum(dim=(-2, -1))
    else:
        tv_per_sample = grad_x.abs().sum(dim=(-2, -1)) + grad_y.abs().sum(dim=(-2, -1))

    if reduction == "mean":
        tv = tv_per_sample.mean()
    elif reduction == "sum":
        tv = tv_per_sample.sum()
    else:
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'mean' or 'sum'.")

    return tv.detach().cpu().item()


def compute_tv_error(pred_depth, gt_depth, reduction="mean"):
    """
    Compute TV/gradient error between predicted and ground-truth depth maps.

    This is the mean absolute difference between forward horizontal and vertical
    gradients of the prediction and ground truth.
    """
    pred = _as_depth_batch(pred_depth)
    gt = _as_depth_batch(gt_depth).to(device=pred.device)

    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and ground truth must have the same shape, got {tuple(pred.shape)} and {tuple(gt.shape)}")

    pred_grad_x, pred_grad_y = _depth_forward_gradients(pred)
    gt_grad_x, gt_grad_y = _depth_forward_gradients(gt)

    error_per_sample = (
        (pred_grad_x - gt_grad_x).abs().mean(dim=(-2, -1))
        + (pred_grad_y - gt_grad_y).abs().mean(dim=(-2, -1))
    )

    if reduction == "mean":
        error = error_per_sample.mean()
    elif reduction == "sum":
        error = error_per_sample.sum()
    else:
        raise ValueError(f"Unsupported reduction '{reduction}'. Use 'mean' or 'sum'.")

    return error.detach().cpu().item()


def compute_residual_total_variation(pred_depth, gt_depth, isotropic=False, reduction="mean"):
    """
    Compute total variation of the absolute residual map |P - G|.
    """
    pred = _as_depth_batch(pred_depth)
    gt = _as_depth_batch(gt_depth).to(device=pred.device)

    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and ground truth must have the same shape, got {tuple(pred.shape)} and {tuple(gt.shape)}")

    residual = (pred - gt).abs()
    return compute_total_variation(residual, isotropic=isotropic, reduction=reduction)


class DepthOutBufferDiagnostics:
    def __init__(self, maxlen=50):
        self.depth_out_history = deque(maxlen=maxlen)

    def update(self, depth_out):
        self.depth_out_history.append(depth_out.detach().cpu())

        if len(self.depth_out_history) < 2:
            mean_per_pixel_std = 0.0
        else:
            depth_history = torch.stack(list(self.depth_out_history), dim=0)
            per_pixel_std = depth_history.std(dim=0, unbiased=False)
            mean_per_pixel_std = per_pixel_std.mean().item()

        return {
            "diag/depth_out_buffer_per_pixel_std_mean": mean_per_pixel_std,
        }

class FeauturesBufferDiagnostics:
    def __init__(self, maxlen=50):
        self.features_history = deque(maxlen=maxlen)
    
    def gemini_center_dominance(self, depth_logits):
        """
        Calculates center dominance for an MDE model using depth bins.
        
        Args:
            depth_logits (torch.Tensor): Shape (B, D, H, W) 
                                        where D is the number of depth bins.
        """
        # 1. Flatten spatial and batch dimensions to treat pixels as individual samples
        # Resulting shape: (N, D)
        B, D, H, W = depth_logits.shape
        u = depth_logits.permute(0, 2, 3, 1).reshape(-1, D)
        
        # 2. Mean vector across all pixels
        u_bar = torch.mean(u, dim=0)
        
        # 3. L2 Norm of the mean vector
        norm_u_bar = torch.norm(u_bar, p=2)
        
        # 4. Average of individual pixel L2 norms
        individual_norms = torch.norm(u, p=2, dim=1)
        e_norm_u = torch.mean(individual_norms)
        
        # 5. Center Dominance Ratio
        return (norm_u_bar / e_norm_u).item()
    
    def calculate_temporal_pixel_dominance(self, feat_history, eps=1e-8):
        """
        Calculates temporal center dominance for B=1.
        
        Args:
            feat_history (torch.Tensor): Shape (HIST_LEN, 1, C, H, W)
        Returns:
            pixel_cd (torch.Tensor): Shape (H, W)
        """
        # 1. Merge HIST_LEN and B: Resulting shape (T, C, H, W)
        # Since B=1, we can just squeeze the batch dimension
        u = feat_history.squeeze(1)
        
        # 2. Calculate the mean feature vector over history (u_bar)
        # Shape: (C, H, W)
        u_bar = torch.mean(u, dim=0)
        
        # 3. Calculate L2 norm of the average vector (||u_bar||_2)
        # Shape: (H, W)
        norm_u_bar = torch.norm(u_bar, p=2, dim=0)
        
        # 4. Calculate individual norms for every vector in history
        # Shape: (T, H, W)
        individual_norms = torch.norm(u, p=2, dim=1)
        
        # 5. Calculate expectation of norms over history (E[||u||_2])
        # Shape: (H, W)
        exp_norm_u = torch.mean(individual_norms, dim=0)
        
        # 6. Center Dominance Ratio: ||Mean(u)|| / Mean(||u||)
        # High values indicate the pixel has "collapsed" to a static state.
        pixel_cd = norm_u_bar / (exp_norm_u + eps)
        
        return pixel_cd

    def update(self, features):
        """
        Only last stage features assumed.
        """
        feat = features.detach().cpu()
        self.features_history.append(feat.detach().cpu())
            
        image_mean_vector_norm = torch.linalg.vector_norm(feat.mean(dim=(0,2,3)), dim=0)
        image_mean_vector_norm = image_mean_vector_norm.unsqueeze(dim=0)
        image_center_dominance = image_mean_vector_norm / torch.linalg.vector_norm(feat.mean(dim=0), dim=0)
        image_mean_center_dominance = image_center_dominance.mean()
            
        gem_image_mean_center_dominance = self.gemini_center_dominance(feat)

        if len(self.features_history) < 2:
            per_pixel_center_dominance = 0.0
            gem_per_pixel_center_dominance = 0.0
            gem_cust_image_mean_center_dominance = 0.0
        else:
            # HIST_LEN,B,C,H,W
            feat_history = torch.stack(list(self.features_history), dim=0)
            
            per_pixel_mean_vector_norm = torch.linalg.vector_norm(
                feat_history.mean(dim=(0, 1)),
                dim=0,
            )
            per_pixel_center_dominance = per_pixel_mean_vector_norm / torch.linalg.vector_norm(feat.mean(dim=0), dim=0)
            per_pixel_center_dominance = per_pixel_center_dominance.mean().item()
            
            gem_per_pixel_center_dominance = self.gemini_center_dominance(feat_history.flatten(start_dim=0,
                                                                                               end_dim=1))
            gem_cust_image_mean_center_dominance = self.calculate_temporal_pixel_dominance(feat_history).mean()

        return {
            # "diag/feat_buffer_per_pixel_center_dominance": per_pixel_center_dominance,
            # "diag/feat_image_mean_center_dominance": image_mean_center_dominance.item(),
            "diag/gem_feat_buffer_per_pixel_center_dominance": gem_per_pixel_center_dominance,
            # "diag/gem_feat_image_mean_center_dominance": gem_image_mean_center_dominance,
            # "diag/gem_cust_feat_image_mean_center_dominance": gem_cust_image_mean_center_dominance,
        }


def compute_image_gradient_chamfer_distance(rgb_image, depth_map, canny_low=50, canny_high=150):
    """
    Compute Image Gradient Chamfer Distance (IGCD).
    
    This measures how closely the structural edges extracted from the depth map 
    align with the edges extracted from the RGB image.
    
    Args:
        rgb_image (torch.Tensor or np.ndarray): RGB image, shape (H, W, 3) or (B, 3, H, W)
        depth_map (torch.Tensor or np.ndarray): Depth map, shape (H, W) or (B, 1, H, W)
        canny_low (int): Lower threshold for Canny edge detector
        canny_high (int): Higher threshold for Canny edge detector
    
    Returns:
        float: Average chamfer distance (lower is better, indicates better alignment)
    """
    # Convert tensors to numpy if needed
    if isinstance(rgb_image, torch.Tensor):
        rgb_np = rgb_image.detach().cpu().numpy()
        if rgb_np.ndim == 4:  # (B, C, H, W)
            rgb_np = rgb_np[0].transpose(1, 2, 0)  # Take first batch, convert to (H, W, C)
        elif rgb_np.ndim == 3 and rgb_np.shape[0] == 3:  # (C, H, W)
            rgb_np = rgb_np.transpose(1, 2, 0)  # Convert to (H, W, C)
    else:
        rgb_np = rgb_image
    
    if isinstance(depth_map, torch.Tensor):
        depth_np = depth_map.detach().cpu().numpy()
        if depth_np.ndim == 4:  # (B, 1, H, W)
            depth_np = depth_np[0, 0]  # Take first batch and channel
        elif depth_np.ndim == 3:  # (1, H, W) or (B, H, W)
            depth_np = depth_np[0] if depth_np.shape[0] == 1 else depth_np
    else:
        depth_np = depth_map
    
    # Ensure rgb is uint8 for Canny
    if rgb_np.dtype != np.uint8:
        rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
    
    # Convert RGB to grayscale for edge detection
    if rgb_np.ndim == 3:
        rgb_gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
    else:
        rgb_gray = rgb_np
    
    # Apply Canny edge detector to RGB
    E_rgb = cv2.Canny(rgb_gray, canny_low, canny_high)
    
    # Normalize depth map and convert to uint8 for Canny
    depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255)
    depth_uint8 = depth_normalized.astype(np.uint8)
    
    # Apply Canny edge detector to depth
    E_depth = cv2.Canny(depth_uint8, canny_low, canny_high)
    
    # Compute distance transform on RGB edges
    # Invert the edge map: distance_transform_edt expects 0s for edges
    dist_transform = distance_transform_edt(E_rgb == 0)
    
    # Get active pixels in depth edge map
    depth_edge_coords = np.where(E_depth > 0)
    
    if len(depth_edge_coords[0]) == 0:
        # No edges detected in depth map
        return float('inf')
    
    # Map depth edges onto the distance transform
    distances = dist_transform[depth_edge_coords]
    
    # Compute average distance
    avg_chamfer_distance = np.mean(distances)
    
    return avg_chamfer_distance


def compute_gradient_magnitude_direction_error(rgb_image, depth_map, edge_threshold=0.1):
    """
    Compute Gradient Magnitude and Direction Error.
    
    This measures how well the gradient directions in the RGB image and depth map align,
    which indicates if the model respects object boundaries.
    
    Args:
        rgb_image (torch.Tensor or np.ndarray): RGB image, shape (H, W, 3) or (B, 3, H, W)
        depth_map (torch.Tensor or np.ndarray): Depth map, shape (H, W) or (B, 1, H, W)
        edge_threshold (float): Threshold for considering a pixel as an edge (percentile-based)
    
    Returns:
        dict: Dictionary containing:
            - 'mean_cosine_similarity': Mean cosine similarity at edge locations
            - 'abs_mean_cosine_similarity': Mean absolute cosine similarity
            - 'alignment_score': Percentage of edges with high alignment (|cos| > 0.7)
    """
    # Convert tensors to numpy if needed
    if isinstance(rgb_image, torch.Tensor):
        rgb_np = rgb_image.detach().cpu().numpy()
        if rgb_np.ndim == 4:  # (B, C, H, W)
            rgb_np = rgb_np[0].transpose(1, 2, 0)  # Take first batch, convert to (H, W, C)
        elif rgb_np.ndim == 3 and rgb_np.shape[0] == 3:  # (C, H, W)
            rgb_np = rgb_np.transpose(1, 2, 0)  # Convert to (H, W, C)
    else:
        rgb_np = rgb_image
    
    if isinstance(depth_map, torch.Tensor):
        depth_np = depth_map.detach().cpu().numpy()
        if depth_np.ndim == 4:  # (B, 1, H, W)
            depth_np = depth_np[0, 0]  # Take first batch and channel
        elif depth_np.ndim == 3:  # (1, H, W)
            depth_np = depth_np[0]
    else:
        depth_np = depth_map
    
    # Convert RGB to grayscale if needed
    if rgb_np.ndim == 3:
        I = cv2.cvtColor(rgb_np.astype(np.float32), cv2.COLOR_RGB2GRAY)
    else:
        I = rgb_np.astype(np.float32)
    
    D = depth_np.astype(np.float32)
    
    # Compute gradients using Sobel filter
    # ∇I = (∂I/∂x, ∂I/∂y)
    grad_I_x = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    grad_I_y = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)
    
    # ∇D = (∂D/∂x, ∂D/∂y)
    grad_D_x = cv2.Sobel(D, cv2.CV_64F, 1, 0, ksize=3)
    grad_D_y = cv2.Sobel(D, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute gradient magnitudes
    mag_I = np.sqrt(grad_I_x**2 + grad_I_y**2)
    mag_D = np.sqrt(grad_D_x**2 + grad_D_y**2)
    
    # Find edge locations (top percentile of gradient magnitudes)
    # We use percentile-based thresholding to adapt to different image contrasts
    threshold_I = np.percentile(mag_I, (1 - edge_threshold) * 100)
    threshold_D = np.percentile(mag_D, (1 - edge_threshold) * 100)
    
    # Edge mask: locations where both RGB and depth have strong gradients
    edge_mask = (mag_I > threshold_I) & (mag_D > threshold_D)
    
    if not np.any(edge_mask):
        # No common edges detected
        return {
            'mean_cosine_similarity': 0.0,
            'abs_mean_cosine_similarity': 0.0,
            'alignment_score': 0.0
        }
    
    # Compute dot product: ∇I · ∇D
    dot_product = grad_I_x * grad_D_x + grad_I_y * grad_D_y
    
    # Compute cosine similarity: (∇I · ∇D) / (||∇I|| * ||∇D||)
    epsilon = 1e-8
    cosine_similarity = dot_product / (mag_I * mag_D + epsilon)
    
    # Extract values at edge locations
    edge_cosine = cosine_similarity[edge_mask]
    
    # Compute metrics
    mean_cos_sim = np.mean(edge_cosine)
    abs_mean_cos_sim = np.mean(np.abs(edge_cosine))
    
    # Alignment score: percentage of edges with |cos| > 0.7 (strong alignment)
    alignment_score = np.mean(np.abs(edge_cosine) > 0.7)
    
    return {
        # 'mean_cosine_similarity': float(mean_cos_sim),
        # 'abs_mean_cosine_similarity': float(abs_mean_cos_sim),
        'alignment_score': float(alignment_score)
    }
