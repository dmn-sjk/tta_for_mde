python adaptation.py \
    --model_name kitti2kitti_ssl_naive_2011_09_26_0086_LR1e-6 \
    --dataset kitti \
    --load_weights_folder ./exp_logs/kitti_sup/models/weights_19 \
    --models_to_load encoder depth \
    --reg_path ./exp_logs/kitti_unsup/models/weights_19 \
    --thres 0.4 \
    --learning_rate 1e-6 \
    --num_workers 0 \
    --data_path /datasets/KITTI \
    --adaptation_method ssl_naive \
    --png \
    --eval_split 2011_09_26_0086 \
    --frame_ids 0 -1 1
    # --scales 0 \
    # --loss_experiment \
    # --dropout
    # --autoblur
    # --amb_masking
    # --frame_ids 0 -1