import torchvision.transforms as transforms
import random
from PIL import ImageFilter


def get_simclr_augs():
    augmentation = [
        # transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        # transforms.ToPILImage(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],
            # [ColorJitterPro(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, gamma=0)], # equivalent to above
            p=0.8,  # not strengthened
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [
                # GaussianBlur([0.1, 2.0])
                transforms.GaussianBlur(kernel_size=13,
                                        sigma=(0.1, 2.0))
            ],
            p=0.5,
        ),
        transforms.RandomHorizontalFlip(),
        # transforms.ToTensor(),
    ]
    return transforms.Compose(augmentation)

class GaussianBlur:
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]) -> None:
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
