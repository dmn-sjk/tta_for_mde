import torchvision
import numpy as np
import torch
from typing import Optional


def get_geom_augs(args, col_augs: Optional[torch.nn.Module] = None):
    transforms = [
        Resize(keep_ratio=True),
        RandomFlip(),
        # dict(type='Normalize', **img_norm_cfg), # assume the images are already normalized form dataloader
    ]

    if col_augs is not None:
        transforms.append(TorchvisionWrapper(col_augs))
    
    return MultiScaleFlipAug(
        img_scale=(args.height, args.width), 
        img_ratios=[
            0.5, 
            0.75,
            1.0, 
            1.25, 
            1.5, 
            1.75, 
            2.0
            ],
        flip=True,
        patch_size=args.patch_size,
        transforms=transforms
    )
    
def svdp_geom_augs(args):
    return MultiScaleFlipAug(
        img_scale=(args.height, args.width), 
        img_ratios=[
            0.5, 
            0.75,
            1.0, 
            1.25, 
            1.5, 
            1.75, 
            2.0
            ],
        flip=True,
        patch_size=args.patch_size,
        transforms=[
            Resize(keep_ratio=True),
            RandomFlip(),
            # dict(type='Normalize', **img_norm_cfg), # assume the images are already normalized form dataloader
        ])

class TorchvisionWrapper:
    def __init__(self, transform: torch.nn.Module):
        self.transform = transform

    def __call__(self, results):
        results['img'] = self.transform(results['img'])
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(\n  {self.transform}\n)"

# based on mmseg MultiScaleFlipAug
class MultiScaleFlipAug:
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        img_scale=(2048, 1024),
        img_ratios=[0.5, 1.0],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]

    After MultiScaleFLipAug with above configuration, the results are wrapped
    into lists of the same length as followed:

    .. code-block::

        dict(
            img=[...],
            img_shape=[...],
            scale=[(1024, 512), (1024, 512), (2048, 1024), (2048, 1024)]
            flip=[False, True, False, True]
            ...
        )

    Args:
        transforms (list[dict]): Transforms to apply in each augmentation.
        img_scale (None | tuple | list[tuple]): Images scales for resizing.
        img_ratios (float | list[float]): Image ratios for resizing
        flip (bool): Whether apply flip augmentation. Default: False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal" and "vertical". If flip_direction is list,
            multiple flip augmentations will be applied.
            It has no effect when flip == False. Default: "horizontal".
    """

    def __init__(self,
                 transforms,
                 img_scale,
                 img_ratios=None,
                 flip=False,
                 flip_direction='horizontal',
                 patch_size=None):
        self.patch_size = patch_size
        self.transforms = torchvision.transforms.Compose(transforms)
        if img_ratios is not None:
            img_ratios = img_ratios if isinstance(img_ratios,
                                                  list) else [img_ratios]
        if img_scale is None:
            # mode 1: given img_scale=None and a range of image ratio
            self.img_scale = None
        elif isinstance(img_scale, tuple):
            assert len(img_scale) == 2
            # mode 2: given a scale and a range of image ratio
            self.img_scale = [(int(img_scale[0] * ratio),
                               int(img_scale[1] * ratio))
                              for ratio in img_ratios]
        else:
            # mode 3: given multiple scales
            self.img_scale = img_scale if isinstance(img_scale,
                                                     list) else [img_scale]

        if img_scale is not None and self.patch_size is not None:
            # NOTE: make the img_scale divisible by patch_size and the result of division by patch size even (needed for the DPT model to work correctly)
            self.img_scale = [(int(img_scale[0] * ratio) // self.patch_size // 2 * self.patch_size * 2,
                               int(img_scale[1] * ratio) // self.patch_size // 2 * self.patch_size * 2)
                              for ratio in img_ratios]
        
        self.flip = flip
        self.img_ratios = img_ratios
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        if not self.flip and self.flip_direction != ['horizontal']:
            print(
                'flip_direction has no effect when flip is set to False')
        if (self.flip
                and not any([isinstance(t, RandomFlip) for t in transforms])):
            print(
                'flip has no effect when RandomFlip is not in transforms')

    def __repr__(self):
        lines = [f"{self.__class__.__name__}("]
        lines.append(f"  img_scale={self.img_scale},")
        lines.append(f"  img_ratios={self.img_ratios},")
        lines.append(f"  flip={self.flip},")
        lines.append(f"  flip_direction={self.flip_direction},")
        lines.append(f"  patch_size={self.patch_size},")
        lines.append(f"  transforms={self.transforms}")
        lines.append(")")
        return "\n".join(lines)

    def __call__(self, x):
        """Call function to apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict[str: list]: The augmented data, where each value is wrapped
               into a list.
        """
        results = {
            'img': x
        }
        aug_data = []
        if self.img_scale is None and self.img_ratios is not None:
            h, w = results['img'].shape[-2:]
            if self.patch_size is not None:
                img_scale = [(int(w * ratio) // self.patch_size // 2 * self.patch_size * 2,
                             int(h * ratio) // self.patch_size // 2 * self.patch_size * 2)
                             for ratio in self.img_ratios]
            else:
                img_scale = [(int(w * ratio), int(h * ratio))
                             for ratio in self.img_ratios]
        else:
            img_scale = self.img_scale
        flip_aug = [False, True] if self.flip else [False]
        for scale in img_scale:
            for flip in flip_aug:
                for direction in self.flip_direction:
                    _results = results.copy()
                    _results['scale'] = scale
                    _results['flip'] = flip
                    _results['flip_direction'] = direction
                    data = self.transforms(_results)
                    aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict
    
class Resize:
    """Resize images & depth.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be Nong, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[-2:]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert all(isinstance(x, tuple) for x in self.img_scale)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates."""
        assert all(isinstance(x, tuple) for x in img_scales)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``."""
        assert all(isinstance(x, tuple) for x in img_scales) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified."""
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``."""
        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[-2:]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            resize = torchvision.transforms.Resize(results['scale'])
            img = resize(results['img'])
            new_h, new_w = img.shape[-2:]
            h, w = results['img'].shape[-2:]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            resize = torchvision.transforms.Resize(results['scale'])
            img = resize(results['img'])
            w_scale = results['scale'][0] / results['img'].shape[-2]
            h_scale = results['scale'][1] / results['img'].shape[-1]

        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"img_scale={self.img_scale}, "
                f"keep_ratio={self.keep_ratio})")

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        return results
    
class RandomFlip:
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"prob={self.prob}, direction='{self.direction}')")

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        if 'flip' not in results:
            flip = True if np.random.rand() < self.prob else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            if results['flip_direction'] == 'horizontal':
                results['img'] = torch.flip(results['img'], dims=[-1])
            else:
                results['img'] = torch.flip(results['img'], dims=[-2])

        return results