# python test.py --data ./data/nuscenes --dataset nuImages --checkpoint_path ./pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth --save_model_path ./log --loss_type L1
# python test.py --data ./data/kitti --dataset kitti --checkpoint_path ./pretrained_models/checkpoint_epoch_390_kitti_L1_best.pth --save_model_path ./log --loss_type L1

python test.py --data ./data/nuscenes --dataset nuImages --checkpoint_path ./pretrained_models/checkpoint_epoch_398_nuimage_L2_best.pth --save_model_path ./log --loss_type L2 

# python test.py --data ./data/kitti --dataset kitti --checkpoint_path ./pretrained_models/checkpoint_epoch_390_kitti_L2_best.pth --save_model_path ./log --loss_type L2
