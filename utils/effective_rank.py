import os
import warnings

import torch
import torch.nn.functional as F


def effective_rank(features):
    # features: (N, dim_feat)
    singular_values = torch.linalg.svdvals(features.float())
    singular_values_sum = torch.linalg.norm(singular_values, ord=1)
    pk = singular_values / singular_values_sum.clamp_min(1e-12)
    entropy = -torch.sum(pk * torch.log(pk.clamp_min(1e-12)))
    return torch.exp(entropy)


def _pool_feature_map(feature):
    if feature.dim() == 4:
        return feature.permute(0, 2, 3, 1).reshape(-1, feature.shape[1])
    if feature.dim() == 3:
        return feature.reshape(-1, feature.shape[-1])
    if feature.dim() == 2:
        return feature

    raise ValueError(f"Unsupported feature shape for effective rank: {tuple(feature.shape)}")


def _normalize_feature_matrix(feature_matrix):
    return F.normalize(feature_matrix.float(), p=2, dim=1)


def _prepare_feature_views(features):
    if isinstance(features, (list, tuple)):
        stage_features = [_normalize_feature_matrix(_pool_feature_map(feature)) for feature in features]
        return {
            "per_stage": stage_features,
            "last": stage_features[-1],
        }

    feature_matrix = _normalize_feature_matrix(_pool_feature_map(features))
    return {
        "per_stage": [feature_matrix],
        "last": feature_matrix,
    }


def _load_holdout(holdout_path, device):
    payload = torch.load(holdout_path, map_location="cpu")

    if isinstance(payload, dict) and "images" in payload:
        holdout = payload["images"]
        if holdout.dtype == torch.uint8:
            holdout = holdout.float() / 255.0

        normalization = payload.get("normalization")
        if normalization is not None:
            mean = normalization["mean"].view(1, -1, 1, 1).float()
            std = normalization["std"].view(1, -1, 1, 1).float()
            holdout = (holdout.float() - mean) / std
    else:
        holdout = payload

    return holdout.to(device)


def _try_load_holdout(holdout_path, device):
    if not holdout_path or not os.path.exists(holdout_path):
        warnings.warn(
            f"Effective rank holdout file is unavailable: {holdout_path}. "
            "Effective rank metrics will be skipped.",
            RuntimeWarning,
        )
        return None

    try:
        return _load_holdout(holdout_path, device)
    except (EOFError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        warnings.warn(
            f"Could not load effective rank holdout from {holdout_path}: {exc}. "
            "Effective rank metrics will be skipped.",
            RuntimeWarning,
        )
        return None


class EffectiveRankHoldout:
    def __init__(
        self,
        holdout_path,
        mean,
        std,
        holdout_every_n=23,
        max_images=50,
    ):
        self.holdout_path = holdout_path
        self.mean = mean.detach().cpu().flatten()
        self.std = std.detach().cpu().flatten()
        self.holdout_every_n = max(1, int(holdout_every_n))
        max_images = int(max_images)
        self.max_images = None if max_images <= 0 else max_images
        self.images = []
        self.global_indices = []
        self.processed_image_count = 0

        os.makedirs(os.path.dirname(self.holdout_path), exist_ok=True)

    def save(self):
        if not self.images:
            return

        torch.save(
            {
                "images": torch.cat(self.images, dim=0),
                "global_indices": torch.cat(self.global_indices, dim=0),
                "normalization": {
                    "mean": self.mean,
                    "std": self.std,
                },
                "holdout_every_n": self.holdout_every_n,
                "max_images": self.max_images,
                "processed_images": self.processed_image_count,
            },
            self.holdout_path,
        )

    def update(self, images):
        current_holdout_size = sum(batch.shape[0] for batch in self.images)
        if self.max_images is not None and current_holdout_size >= self.max_images:
            self.processed_image_count += images.shape[0]
            return

        batch_size = images.shape[0]
        image_indices = torch.arange(
            self.processed_image_count,
            self.processed_image_count + batch_size,
            device=images.device,
        )
        selected_mask = image_indices % self.holdout_every_n == 0
        self.processed_image_count += batch_size

        if not selected_mask.any():
            return

        selected_images = images[selected_mask]
        selected_indices = image_indices[selected_mask].detach().cpu()

        if self.max_images is not None:
            remaining_capacity = self.max_images - current_holdout_size
            if remaining_capacity <= 0:
                return
            selected_images = selected_images[:remaining_capacity]
            selected_indices = selected_indices[:remaining_capacity]

        if selected_images.numel() == 0:
            return

        selected_images = (
            selected_images.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .cpu()
        )

        self.images.append(selected_images)
        self.global_indices.append(selected_indices)
        self.save()


class EffectiveRank:
    def __init__(
        self,
        holdout_path,
        device,
        feature_stage_mode="all",
    ):
        self.holdout = _try_load_holdout(holdout_path, device)
        self.feature_stage_mode = feature_stage_mode

    def get_effective_rank(self, encoder, feature_stage_mode=None, return_details=False):
        feature_stage_mode = feature_stage_mode or self.feature_stage_mode
        if feature_stage_mode not in {"all", "last"}:
            raise ValueError(
                f"Unsupported feature_stage_mode '{feature_stage_mode}'. Expected 'all' or 'last'."
            )

        if self.holdout is None:
            metrics = {
                "all": None,
                "last": None,
                "per_stage": [],
            }
            return metrics if return_details else metrics[feature_stage_mode]

        train_mask = [module.training for module in encoder.modules()]
        encoder.eval()

        with torch.no_grad():
            features = encoder(self.holdout)
            feature_views = _prepare_feature_views(features)

        per_stage_ranks = [
            effective_rank(stage_feature).cpu().item()
            for stage_feature in feature_views["per_stage"]
        ]
        metrics = {
            "all": sum(per_stage_ranks) / len(per_stage_ranks),
            "last": effective_rank(feature_views["last"]).cpu().item(),
            "per_stage": per_stage_ranks,
        }

        if any(train_mask):
            for module, was_training in zip(encoder.modules(), train_mask):
                if was_training:
                    module.train()

        if return_details:
            return metrics

        return metrics[feature_stage_mode]