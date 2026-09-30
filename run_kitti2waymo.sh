python adaptation.py \
    --model_name kitti2waymo_ssl_naive \
    --dataset waymo \
    --load_weights_folder ./exp_logs/kitti_sup/models/weights_19 \
    --models_to_load encoder depth \
    --reg_path ./exp_logs/kitti_unsup/models/weights_19 \
    --thres 0.4 \
    --learning_rate 1e-5 \
    --num_workers 0 \
    --data_path /datasets/waymo \
    --png \
    --adaptation_method ssl_naive
    # --gt_transform
    # --frame_ids 0 -1 \
