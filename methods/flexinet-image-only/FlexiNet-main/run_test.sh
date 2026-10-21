python test.py --mode nuscenes --config configs/test_chassis_aligned.yaml

# Legacy examples:
# python test.py --mode nuscenes --config configs/test.yaml
# python test.py --mode legacy --data ./data/nuscenes --dataset nuImages --checkpoint_path ./pretrained_models/checkpoint_epoch_100_nuimage_L1_best.pth --loss_type L1
# python test.py --mode legacy --data ./data/kitti --dataset kitti --checkpoint_path ./pretrained_models/checkpoint_epoch_390_kitti_L1_best.pth --loss_type L1
# python test.py --mode legacy --data ./data/kitti --dataset kitti --checkpoint_path ./pretrained_models/checkpoint_epoch_390_kitti_L2_best.pth --loss_type L2
