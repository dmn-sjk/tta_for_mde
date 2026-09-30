import numpy as np
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from torch.utils.data import Dataset
import os


class RoboDepthDataset(Dataset):
    
    def __init__(self, data_dir, image_list, H, W):
        self.image_list = image_list
        with open(self.image_list, "r") as f:
            self.total_images = f.read().splitlines()
        self.data_dir = data_dir
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}
        
        saved_folders = []
        all_imgs = []
        # adjust path and take all images from the given sequence
        for img in self.total_images:
            path, idx, side = img.split(' ') 
            folder = os.path.join(self.data_dir, path, f"image_0{self.side_map[side]}", 'data')
            if folder in saved_folders:
                continue
            
            all_imgs_in_folder = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
            all_imgs.extend(all_imgs_in_folder)
            saved_folders.append(folder)
                                  
        self.total_images = all_imgs
            
        self.H = H
        self.W = W
        

        # self.total_images = self.total_images[197:]
        print(len(self.total_images))
    
    def __len__(self):
        return len(self.total_images)

    def __getitem__(self, idx):
        # kitti_data/2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000054.png
        
        image_loc = self.total_images[idx]
        image = Image.open(image_loc).convert("RGB")  # [1242, 375], original size
        image = image.resize((self.W, self.H), Image.ANTIALIAS)  # [640, 192], resized
        image = np.asarray(image)
        loc_return = os.path.join('kitti_data', *image_loc.split('/')[-5:])
        return (image, loc_return)

