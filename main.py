import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)


import warnings
import argparse
from ultralytics import YOLO

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO Training/Validation/Prediction Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', type=str, default='./cfg/models/YOLO11-AU-IR.yaml', help='Model configuration file (.yaml)')
    train_parser.add_argument('--data', type=str, default='./cfg/datasets/AUVD-Seg300.yaml', help='Dataset configuration file (.yaml)')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--workers', type=int, default=0, help='Number of workers')
    train_parser.add_argument('--device', type=str, default=None, help='Device to use (e.g. 0 or 0,1,2,3)')
    train_parser.add_argument('--optimizer', type=str, default='SGD', help='Optimizer to use')
    train_parser.add_argument('--project', type=str, default='runs/train', help='Project directory')
    train_parser.add_argument('--name', type=str, default='exp', help='Experiment name')
    train_parser.add_argument('--cache', action='store_true', help='Use cache')
    train_parser.add_argument('--close-mosaic', type=int, default=0, help='Close mosaic augmentation')

    # Val command
    val_parser = subparsers.add_parser('val', help='Validate a model')
    val_parser.add_argument('--model', type=str, required=True, help='Model weights file (.pt)')
    val_parser.add_argument('--data', type=str, required=True, help='Dataset configuration file (.yaml)')
    val_parser.add_argument('--split', type=str, default='val', help='Dataset split to use')
    val_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    val_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    val_parser.add_argument('--project', type=str, default='runs/val', help='Project directory')
    val_parser.add_argument('--name', type=str, default='exp', help='Experiment name')

    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Run prediction')
    pred_parser.add_argument('--model', type=str, required=True, help='Model weights file (.pt)')
    pred_parser.add_argument('--source', type=str, required=True, help='Source for prediction')
    pred_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    pred_parser.add_argument('--project', type=str, default='runs/seg', help='Project directory')
    pred_parser.add_argument('--name', type=str, default='AUIR', help='Experiment name')
    pred_parser.add_argument('--save', action='store_true', help='Save results')
    pred_parser.add_argument('--visualize', action='store_true', help='Visualize features')

    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.command == 'train':
        model = YOLO(args.model)
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            device=args.device,
            optimizer=args.optimizer,
            project=args.project,
            name=args.name,
            cache=args.cache,
            close_mosaic=args.close_mosaic
        )
    elif args.command == 'val':
        model = YOLO(args.model)
        model.val(
            data=args.data,
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name
        )
    elif args.command == 'predict':
        model = YOLO(args.model)
        model.predict(
            source=args.source,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            save=args.save,
            visualize=args.visualize,
        )
    else:
        raise ValueError(f"Unknown command: {args.command}")

if __name__ == '__main__':
    main()