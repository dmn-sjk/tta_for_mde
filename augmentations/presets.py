"""
Augmentation registry, config builder, and presets.

Supports two ways of defining an augmentation pipeline:
1. **Preset name** — a built-in config looked up from ``PRESETS``.
2. **YAML config file** — with full parameter control for both photometric
   and geometric augmentations.

YAML config format
------------------

**Photo-only** (backward-compatible list):

.. code-block:: yaml

    - clip:
        min_val: 0.0
        max_val: 1.0
    - color_jitter:
        brightness: [0.6, 1.4]
        contrast: [0.7, 1.3]
    - gaussian_blur:
        kernel_size: 5
        sigma: [0.001, 0.5]
    - gaussian_noise:
        std: 0.005
        clip: true

**Photo + Geom** (dict with ``photo`` and ``geom`` keys):

.. code-block:: yaml

    photo:
      - clip:
          min_val: 0.0
          max_val: 1.0
      - color_jitter:
          brightness: [0.6, 1.4]

    geom:
      img_ratios: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
      flip: true

The ``img_scale`` and ``patch_size`` for ``MultiScaleFlipAug`` are resolved
at runtime from the existing ``--height`` / ``--width`` / model-specific
``patch_size`` — they are **not** set in the augmentation config.

Wrapper augmentations (``random_apply``) use a nested ``transforms`` key:

.. code-block:: yaml

    - random_apply:
        p: 0.8
        transforms:
          - color_jitter:
              brightness: 0.8
              contrast: 0.8
"""

from __future__ import annotations

import yaml
from typing import Optional, List, Dict, Union
from torchvision import transforms as T

from augmentations.photo import (
    Clip,
    ColorJitter,
    GaussianBlur,
    GaussianNoise,
    RandomGrayscale,
)
from augmentations.geom import MultiScaleFlipAug, Resize, RandomFlip, TorchvisionWrapper

# ---------------------------------------------------------------------------
# Registry: name → class
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, type] = {
    "clip": Clip,
    "color_jitter": ColorJitter,
    "gaussian_blur": GaussianBlur,
    "gaussian_noise": GaussianNoise,
    "random_grayscale": RandomGrayscale,
    # "random_horizontal_flip": T.RandomHorizontalFlip,
}

# Names that receive special (wrapper) handling
_WRAPPERS = {"random_apply"}


def register(name: str, cls: type) -> None:
    """Add a new augmentation to the registry at runtime."""
    REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_aug(spec: dict) -> T.transforms.Transform:
    """Instantiate a single augmentation from a *spec* dict.

    A spec has exactly one key (the augmentation name) whose value is either
    ``None`` (no params) or a dict of keyword arguments.

    Wrapper augmentations (e.g. ``random_apply``) may contain a nested
    ``transforms`` list that is recursively built.
    """
    if len(spec) != 1:
        raise ValueError(
            f"Each augmentation spec must have exactly one key, got: {list(spec.keys())}"
        )

    name, params = next(iter(spec.items()))
    params = params or {}

    # --- wrapper: random_apply ---
    if name == "random_apply":
        inner_specs = params.pop("transforms", [])
        inner = [build_aug(s) for s in inner_specs]
        return T.RandomApply(inner, **params)

    # --- regular augmentation ---
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown augmentation: '{name}'. "
            f"Registered: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name](**params)


def build_photo_augs(config: List[Dict]) -> T.Compose:
    """Build a ``torchvision.transforms.Compose`` pipeline from a config list."""
    return T.Compose([build_aug(spec) for spec in config])


def build_geom_augs(
    args,
    geom_config: Dict,
    photo_augs: Optional[T.Compose] = None,
) -> MultiScaleFlipAug:
    """Build a ``MultiScaleFlipAug`` geometric augmentation wrapper.

    Args:
        geom_config: Dict with keys ``img_ratios`` (list[float]) and
            ``flip`` (bool).
        photo_augs: Optional photometric pipeline to apply inside the
            geometric wrapper (via ``TorchvisionWrapper``).
    """
    img_ratios = geom_config.get("img_ratios", [1.0])
    flip = geom_config.get("flip", False)

    transforms = [
        Resize(keep_ratio=True),
        RandomFlip(),
    ]
    if photo_augs is not None:
        transforms.append(TorchvisionWrapper(photo_augs))

    return MultiScaleFlipAug(
        img_scale=(args.height, args.width), 
        img_ratios=img_ratios,
        flip=flip,
        transforms=transforms,
        patch_size=args.patch_size,
    )


# ---------------------------------------------------------------------------
# YAML I/O
# ---------------------------------------------------------------------------

def load_config(path: str) -> Union[Dict, List[Dict]]:
    """Load an augmentation config from a YAML file.

    Returns either a plain list (photo-only config) or a dict with ``photo``
    and/or ``geom`` keys.
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if isinstance(config, list):
        return config
    if isinstance(config, dict):
        if "photo" not in config and "geom" not in config:
            raise ValueError(
                "Dict-style config must contain a 'photo' and/or 'geom' key. "
                f"Got keys: {list(config.keys())}"
            )
        return config
    raise ValueError(
        f"Augmentation config must be a YAML list or dict, got {type(config).__name__}"
    )


# ---------------------------------------------------------------------------
# Built-in presets (same format as YAML configs)
# ---------------------------------------------------------------------------

# PRESETS: dict[str, list[dict]] = {
#     "cotta": [
#         {"clip": {"min_val": 0.0, "max_val": 1.0}},
#         {
#             "color_jitter": {
#                 "brightness": [0.6, 1.4],
#                 "contrast": [0.7, 1.3],
#                 "saturation": [0.5, 1.5],
#                 "hue": [-0.06, 0.06],
#                 "gamma": [0.7, 1.3],
#             }
#         },
#         {"gaussian_blur": {"kernel_size": 5, "sigma": [0.001, 0.5]}},
#         {"gaussian_noise": {"mean": 0, "std": 0.005, "clip": True}},
#     ],
#     "cotta_soft": [
#         {"clip": {"min_val": 0.0, "max_val": 1.0}},
#         {
#             "color_jitter": {
#                 "brightness": [0.8, 1.2],
#                 "contrast": [0.85, 1.15],
#                 "saturation": [0.75, 1.25],
#                 "hue": [-0.03, 0.03],
#                 "gamma": [0.85, 1.15],
#             }
#         },
#         {"gaussian_blur": {"kernel_size": 5, "sigma": [0.001, 0.25]}},
#         {"gaussian_noise": {"mean": 0, "std": 0.005, "clip": True}},
#     ],
#     "sim_clr": [
#         {
#             "random_apply": {
#                 "p": 0.8,
#                 "transforms": [
#                     {
#                         "color_jitter": {
#                             "brightness": 0.8,
#                             "contrast": 0.8,
#                             "saturation": 0.8,
#                             "hue": 0.2,
#                         }
#                     },
#                 ],
#             }
#         },
#         {"random_grayscale": {"p": 0.2}},
#         {
#             "random_apply": {
#                 "p": 0.5,
#                 "transforms": [
#                     {"gaussian_blur": {"kernel_size": 13, "sigma": [0.1, 2.0]}},
#                 ],
#             }
#         },
#         geom:
#         img_ratios: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
#         flip: true
#         # {"random_horizontal_flip": {}},
#     ],
# }


# def get_preset(name: str) -> T.Compose:
#     """Build an augmentation pipeline from a built-in preset name.

#     Raises:
#         ValueError: If *name* is not in ``PRESETS``.
#     """
#     if name not in PRESETS:
#         raise ValueError(
#             f"Unknown preset: '{name}'. Available: {list(PRESETS.keys())}"
#         )
#     return build_photo_augs(PRESETS[name])
