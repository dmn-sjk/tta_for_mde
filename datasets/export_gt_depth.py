import os
import sys
# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np
import PIL.Image as pil
from utils.utils import readlines
from kitti_utils import generate_depth_map
import os.path as osp
# from pyquaternion import Quaternion
# from datasets.nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
# from datasets.nuscenes.nuscenes import NuScenes
# from datasets.nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

def export_gt_depths():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        required=True,
                        choices=["eigen", "eigen_benchmark", "eigen_zhou", "nuScenes_test", "nuScenes_val", "nuScenes_test_night",
                                "cloudy", "foggy", "rainy", "sunny"])
    opt = parser.parse_args()

    #eigen_zhou represents validation splits
    current = os.getcwd()

    if opt.split in ["eigen", "eigen_benchmark", "nuScenes_test"]:
        split_folder = os.path.join(current, "splits", opt.split)
        lines = readlines(os.path.join(split_folder, "test_files.txt"))
    elif opt.split in ["cloudy", "foggy", "rainy", "sunny"]:
        split_folder = os.path.join(current, "splits", 'driving_stereo', opt.split)
        lines = readlines(os.path.join(split_folder, "test_files.txt"))
    elif opt.split == "nuScenes_test_night":
        split_folder = os.path.join(current, "splits", opt.split)
        lines = readlines(os.path.join(split_folder, "night_lidar.txt"))
    else:
        split_folder = os.path.join(current, "splits", opt.split)
        lines = readlines(os.path.join(split_folder, "val_files.txt"))

    # if opt.split == "nuScenes_test":
    #     nusc = NuScenes(version="v1.0-test", dataroot=opt.data_path, verbose=False)
    # else:
    #     nusc = NuScenes(version='v1.0-trainval', dataroot=opt.data_path, verbose=False)

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in tqdm(lines):

        if opt.split == "nuScenes_val":
            cam_index, lidar_index = line.split()
        elif opt.split =="nuScenes_test" or opt.split =="nuScenes_test_night":
            lidar_index = line
        elif opt.split == "eigen" or opt.split == "eigen_benchmark":
            folder, frame_id, _, type = line.split()
            frame_id = int(frame_id)
        elif opt.split == "cloudy" or opt.split == "foggy" or opt.split == "rainy" or opt.split == "sunny":
            pass
        else:
            folder, frame_id, _ = line.split()
            frame_id = int(frame_id)

        if opt.split == "eigen":
            calib_dir = os.path.join(current, opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(current, opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split == "eigen_benchmark":
            gt_depth_path = os.path.join(current,opt.data_path, folder, "proj_depth",
                                         "groundtruth", "image_02", "{:010d}.png".format(frame_id))
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
        elif opt.split == "eigen_zhou":
            calib_dir = os.path.join(current,opt.data_path, folder.split("/")[0])
            velo_filename = os.path.join(current,opt.data_path, folder,
                                         "velodyne_points/data", "{:010d}.bin".format(frame_id))
            gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True)
        elif opt.split in ["nuScenes_test", "nuScenes_val", "nuScenes_test_night"]:
            if opt.split == "nuScenes_test" or opt.split == "nuScenes_test_night":
                sample = nusc.sample[int(lidar_index)]
                pointsensor_token = sample['data']['LIDAR_TOP']
                camsensor_token = sample['data']["CAM_FRONT"]
            else:
                sample_lidar = nusc.sample_data[int(lidar_index)]
                sample_cam = nusc.sample_data[int(cam_index)]
                pointsensor_token = sample_lidar
                camsensor_token = sample_cam
            
            gt_depth = get_depth(nusc, pointsensor_token, camsensor_token)
           
        elif opt.split == "sunny" or opt.split == "foggy" or opt.split == "rainy" or opt.split == "cloudy":
            gt_depth_path = os.path.join(opt.data_path, opt.split, opt.split, 'depth-map-full-size', line)
            gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 256
            
            
        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths()
