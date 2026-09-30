python adaptation.py \
    --model_name kitti2kitti_sslnaive \
    --dataset kitti_depth \
    --load_weights_folder ./exp_logs/kitti_sup/models/weights_19 \
    --models_to_load encoder depth \
    --reg_path ./exp_logs/kitti_unsup/models/weights_19 \
    --thres 0.4 \
    --learning_rate 1e-5 \
    --num_workers 0 \
    --data_path /datasets/KITTI \
    --adaptation_method ssl_naive \
    --png
    # --frame_ids 0 -1 