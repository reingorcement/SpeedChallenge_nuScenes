# python train.py --data ./data/nuscenes --dataset nuImages --save_model_path ./log/nuscenes_flexinet_model --loss_type L1
python train.py --data ./data/nuscenes --dataset nuImages --save_model_path ./log/nuscenes_flexinet_model --loss_type L2 --use_amp
# python train.py --data ./data/kitti --dataset kitti --save_model_path ./log/kitti_flexinet_model --loss_type L1
# python train.py --data ./data/kitti --dataset kitti --save_model_path ./slog/kitti_flexinet_model --loss_type L2
