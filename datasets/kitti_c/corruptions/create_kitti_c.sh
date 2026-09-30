python3 create.py --image_list /net/people/plgrid/plgdmnsjk/ada-depth/splits/eigen_benchmark/val_files_bak.txt \
    --H 352 \
    --W 1216 \
    --save_path /net/tscratch/people/plgdmnsjk/datasets/KITTI/kitti_c/kitti_c_1216x352 \
    --data_dir /net/tscratch/people/plgdmnsjk/datasets/KITTI/kitti_data \
    --if_brightness true \
    --if_fog \
    --if_contrast \
    --if_defocus_blur \
    --if_motion_blur \
    --if_elastic \
    --if_gaussian_noise \
    --if_impulse_noise \
    --if_shot_noise \
    --if_jpeg 
    # --if_impulse_noise