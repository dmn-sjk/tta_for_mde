from augmentations.presets import (
    build_photo_augs,
    build_geom_augs,
    load_config,
    REGISTRY,
)
import os
from typing import Optional, List, Dict


def get_augs(args):
    """Build an augmentation pipeline from CLI arguments.

    Supports two modes (``--augs_config`` takes priority over ``--augs``):

    1. ``--augs_config path/to/config.yaml``  — full control over which
       augmentations to compose and all their parameters.  The YAML can be a
       plain list (photo-only) or a dict with ``photo`` / ``geom`` sections.
    2. ``--augs <preset>``  — use a built-in preset name
       (e.g. ``cotta``, ``cotta_soft``, ``sim_clr``).

    Geometric augmentations are enabled when:
    - the YAML config contains a ``geom`` section, **or**
    - the ``--geom_augs`` flag is set on the command line.

    When geometric augs are active the photometric pipeline is wrapped inside
    a :class:`~augmentations.geom.MultiScaleFlipAug` that handles
    multi-scale resize + flip.

    Returns:
        A ``torchvision.transforms.Compose`` (photo-only) or a
        ``MultiScaleFlipAug`` (photo + geom), or ``None`` if no augmentation
        flags were provided.
    """
    config_name = getattr(args, "augs_config", None)
    config_path = os.path.join('configs', 'augmentations', config_name + '.yaml')
    # preset_name = getattr(args, "augs", None)

    # ------------------------------------------------------------------
    # 1. Determine the photometric config  (list[dict] | None)
    #    and the geometric config            (dict | None)
    # ------------------------------------------------------------------
    photo_config: Optional[List[Dict]] = None
    geom_config: Optional[Dict] = None

    if config_path is not None:
        raw = load_config(config_path)
        if isinstance(raw, list):
            photo_config = raw
        else:                          # dict with 'photo' / 'geom' keys
            photo_config = raw.get("photo")
            geom_config = raw.get("geom")
    # elif preset_name is not None:
    #     photo_config = PRESETS.get(preset_name)
    #     if photo_config is None:
    #         raise ValueError(
    #             f"Unknown preset: '{preset_name}'. "
    #             f"Available: {list(PRESETS.keys())}"
    #         )

    # CLI --geom_augs flag (creates a default geom config if not from YAML)
    use_geom = getattr(args, "geom_augs", False)
    if use_geom and geom_config is None:
        geom_config = {}

    # CLI overrides for geom parameters
    if geom_config is not None:
        cli_ratios = getattr(args, "img_ratios", None)
        cli_flip = getattr(args, "flip_augs", None)
        if cli_ratios is not None:
            geom_config.setdefault("img_ratios", cli_ratios)
        if cli_flip is not None:
            geom_config.setdefault("flip", cli_flip)

    # ------------------------------------------------------------------
    # 2. Build the pipelines
    # ------------------------------------------------------------------
    photo_augs = build_photo_augs(photo_config) if photo_config else None

    if geom_config is not None:
        result = build_geom_augs(
            args,
            geom_config,
            photo_augs=photo_augs,
        )
    else:
        result = photo_augs
        
    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    print(result)
    # _print_summary(result, photo_config, geom_config)

    return result


def _print_summary(pipeline, photo_config, geom_config):
    """Print a human-readable summary of the augmentation pipeline."""
    if pipeline is None:
        print("[Augmentations] None")
        return

    lines = ["[Augmentations]"]

    if photo_config:
        lines.append("  Photometric:")
        for spec in photo_config:
            name, params = next(iter(spec.items()))
            if name == "random_apply":
                inner_names = ", ".join(
                    next(iter(s.keys())) for s in params.get("transforms", [])
                )
                lines.append(f"    - {name}(p={params.get('p', 0.5)}, [{inner_names}])")
            elif params:
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"    - {name}({param_str})")
            else:
                lines.append(f"    - {name}()")

    if geom_config:
        lines.append("  Geometric (MultiScaleFlipAug):")
        for k, v in geom_config.items():
            lines.append(f"    - {k}: {v}")

    print("\n".join(lines))
