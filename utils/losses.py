import torch
import torch.nn as nn


class DepthLoss(nn.Module):
    """ Depth Loss """
    def __init__(self, loss='l1', ignore_index = 0):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = torch.nn.L1Loss(reduction='none')
        elif loss == 'mse':
            self.loss = torch.nn.MSELoss(reduction='none')
        elif loss == 'silog':
            self.loss = SilogLoss()
        else:
            raise ValueError('Loss %s currently not supported' %(self.loss))

        self.ignore_index = ignore_index

    def forward(self, prediction, ground_truth):
        mask = (ground_truth != self.ignore_index)
        loss_image = torch.full_like(ground_truth, float('nan'))  # Initialize loss with NaNs
        computed_loss = self.loss(torch.masked_select(prediction, mask), torch.masked_select(ground_truth, mask))
        loss_image[mask] = computed_loss  # Apply computed loss to non-ignored positions
        return loss_image, computed_loss.mean()


class SilogLoss(nn.Module):
    def __init__(self, variance_focus=1.0):
        super(SilogLoss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt):
        d = torch.log(depth_est) - torch.log(depth_gt)
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


class TripletLoss(nn.Module):
    """
    Triplet Loss for feature learning
    
    Takes anchor, positive, and negative tensors of shape (N, C) where:
    - N is the number of samples
    - C is the feature dimensionality
    - anchor[i] corresponds to positive[i] and negative[i]
    """
    def __init__(self, margin=1.0, reduction='mean'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        
    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: Tensor of shape (N, C) - anchor samples
            positive: Tensor of shape (N, C) - positive samples corresponding to anchors
            negative: Tensor of shape (N, C) - negative samples corresponding to anchors
        
        Returns:
            loss: Triplet loss value
        """
        # Compute distances
        dist_pos = torch.sum((anchor - positive) ** 2, dim=1)  # distance between anchor and positive
        dist_neg = torch.sum((anchor - negative) ** 2, dim=1)  # distance between anchor and negative
        
        # Compute triplet loss with margin
        losses = torch.clamp(dist_pos - dist_neg + self.margin, min=0.0)

        # Apply reduction
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise ValueError(f"Reduction '{self.reduction}' not supported. Use 'mean', 'sum', or 'none'.")

