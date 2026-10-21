from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.nuscenes_vision_eval import NuScenesVisionEvalDataset
from model.FlexiNet import FlexiNet
from utils.checkpoint import load_model_checkpoint
from utils.evaluation import build_criterion, copy_small_results_for_git, run_eval, write_evaluation_artifacts
from utils.plotting import plot_true_vs_estimated_from_csv
from utils.seed import set_seed


PROJECT_ROOT = Path(__file__).resolve().parent


def load_config_from_yaml(path: Path) -> dict:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for config loading. Install it with: pip install PyYAML") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(value: str | Path) -> Path:
    value_str = str(value)
    if os.name == "nt" and value_str.startswith("/") and not value_str.startswith("//"):
        return Path(value_str)
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def resolve_config_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.is_file():
        return cwd_path.resolve()
    return (PROJECT_ROOT / path).resolve()


def run_nuscenes_baseline(params: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Pure-vision FlexiNet nuScenes evaluation")
    parser.add_argument("--config", type=str, default="configs/test_chassis_aligned.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--dataroot", type=str, default=None)
    parser.add_argument("--can_bus_root", type=str, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--first_n", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--cpu", action="store_true", help="Force CPU evaluation")
    parser.add_argument(
        "--include_predictions_in_results",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy val_predictions.csv into results_dir for later analysis.",
    )
    args = parser.parse_args(params)

    cfg = load_config_from_yaml(resolve_config_path(args.config))
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["eval"]

    checkpoint_path = resolve_project_path(args.checkpoint or model_cfg["checkpoint_path"])
    dataroot = resolve_project_path(args.dataroot or data_cfg["dataroot"])
    can_bus_root = resolve_project_path(args.can_bus_root or data_cfg["can_bus_root"])
    output_dir = resolve_project_path(args.output_dir or eval_cfg["output_dir"])
    results_dir = resolve_project_path(args.results_dir or eval_cfg["results_dir"])
    split = args.split or data_cfg["split"]
    prefix = args.prefix or eval_cfg["prefix"]
    first_n = int(args.first_n if args.first_n is not None else eval_cfg["first_n"])
    batch_size = int(args.batch_size if args.batch_size is not None else eval_cfg["batch_size"])
    num_workers = int(args.num_workers if args.num_workers is not None else eval_cfg["num_workers"])

    set_seed(int(eval_cfg["seed"]))
    use_gpu = bool(eval_cfg.get("use_gpu", True)) and not args.cpu
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"nuScenes dataroot: {dataroot}")
    print(f"CAN bus root for scene alignment: {can_bus_root}")

    dataset = NuScenesVisionEvalDataset(
        dataroot=dataroot,
        version=data_cfg["version"],
        split=split,
        img_size=(int(data_cfg["img_height"]), int(data_cfg["img_width"])),
        seq_len=int(data_cfg["seq_len"]),
        cam_channel=data_cfg["cam_channel"],
        can_bus_root=can_bus_root,
        filter_can_bus_scenes=bool(data_cfg.get("filter_can_bus_scenes", True)),
        image_cache_size=int(data_cfg.get("image_cache_size", 0)),
    )
    print(f"Eval scenes: {len(dataset.scene_names)} | Eval samples: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = FlexiNet(
        input_channels=int(model_cfg.get("input_channels", 1)),
        num_classes=int(model_cfg.get("num_classes", 1)),
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    load_model_checkpoint(model, checkpoint_path, device)

    criterion = build_criterion(str(eval_cfg.get("loss", "l1")))
    result = run_eval(model, loader, criterion, device)
    metrics = write_evaluation_artifacts(result, output_dir, prefix=prefix)
    plot_true_vs_estimated_from_csv(
        output_dir / f"{prefix}_predictions.csv",
        output_dir,
        prefix=prefix,
        first_n=first_n,
    )
    copy_small_results_for_git(
        output_dir,
        results_dir,
        include_predictions=bool(args.include_predictions_in_results),
    )

    print(f"Loss: {metrics['loss']:.4f}")
    print(f"MAE: {metrics['mae_mps']:.4f} m/s ({metrics['mae_kmh']:.4f} km/h)")
    print(f"RMSE: {metrics['rmse_mps']:.4f} m/s ({metrics['rmse_kmh']:.4f} km/h)")
    print(f"R2: {metrics['r2']:.6f}")
    print(f"Saved evaluation artifacts to: {output_dir}")
    print(f"Copied small analysis artifacts to: {results_dir}")


def run_legacy_test(params: list[str]) -> None:
    from dataset.kitti import KITTIDataset
    from dataset.nuimages import NuImagesDataset

    parser = argparse.ArgumentParser(description="Legacy FlexiNet KITTI/nuImages testing")
    parser.add_argument("--data", type=str, required=True, help="Path of test data")
    parser.add_argument("--dataset", type=str, default="Kitti", help="Dataset to use: Kitti, nuImages")
    parser.add_argument("--crop_height", type=int, default=64, help="Height of cropped input image")
    parser.add_argument("--crop_width", type=int, default=64, help="Width of cropped input image")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for testing")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--cuda", type=str, default="0", help="GPU id(s) to use")
    parser.add_argument("--use_gpu", action="store_true", default=True, help="Whether to use GPU")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to pretrained model")
    parser.add_argument("--loss_type", type=str, choices=["L1", "L2"], required=True, help="Loss function type")
    args = parser.parse_args(params)

    set_seed(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    print(f"Using device: {args.device}")

    dataset_name = args.dataset.lower()
    if dataset_name == "kitti":
        dataset_test = KITTIDataset(
            dataroot=args.data,
            split="valid",
            img_size=(args.crop_height, args.crop_width),
        )
    elif dataset_name == "nuimages":
        dataset_test = NuImagesDataset(
            dataroot=args.data,
            version="v1.0-test",
            img_size=(args.crop_height, args.crop_width),
            subset=False,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Choose from 'Kitti' or 'nuImages'.")

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = FlexiNet(input_channels=1, num_classes=1)
    if torch.cuda.device_count() > 1 and args.use_gpu:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for testing.")

    model = model.to(args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    load_model_checkpoint(model, args.checkpoint_path, args.device)
    test_loss = legacy_eval(args, model, dataloader_test)
    print(f"Test Loss: {test_loss:.4f}")


def legacy_eval(args: argparse.Namespace, model: nn.Module, dataloader_test: DataLoader) -> float:
    model.eval()
    running_test_loss = 0.0
    eps = 1e-6
    if args.loss_type == "L1":
        criterion = nn.L1Loss()
    elif args.loss_type == "L2":
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid loss type specified. Choose either 'L1' or 'L2'.")

    with torch.no_grad():
        for img_batch, speed in tqdm(dataloader_test, desc="Testing", unit="batch"):
            img_batch = img_batch.to(args.device)
            speed = speed.to(args.device)
            predicted_speed = model(img_batch)
            if args.loss_type == "L1":
                loss = criterion(predicted_speed, speed.unsqueeze(1))
            else:
                mse_loss = criterion(predicted_speed, speed.unsqueeze(1))
                loss = torch.sqrt(mse_loss + eps)
            running_test_loss += loss.item()

    return running_test_loss / max(len(dataloader_test), 1)


def main(params: list[str]) -> None:
    if params and params[0] == "--mode":
        if len(params) < 2:
            raise ValueError("--mode requires either 'nuscenes' or 'legacy'.")
        mode = params[1]
        remaining = params[2:]
    else:
        mode = "nuscenes"
        remaining = params

    if mode == "nuscenes":
        run_nuscenes_baseline(remaining)
        return
    if mode == "legacy":
        run_legacy_test(remaining)
        return
    raise ValueError(f"Unsupported mode: {mode}. Choose 'nuscenes' or 'legacy'.")


if __name__ == "__main__":
    main(sys.argv[1:])
