import argparse
import os
import sys
from pathlib import Path

# Define the path to the yolov5 directory
yolov5_dir = Path(__file__).resolve().parents[1]  # Assuming this is the yolov5 directory

if str(yolov5_dir) not in sys.path:
    sys.path.append(str(yolov5_dir))

# Modify sys.modules to use our custom ROOT
import types
import yolov5
yolov5.train = types.ModuleType('train')
sys.modules['yolov5.train'] = yolov5.train

# Now import train
from yolov5 import train

# Set ROOT to an absolute path
train.ROOT = Path(yolov5.__file__).resolve().parent

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--data', type=str, default='D:/smartparking/pklot_dataset/pklot_dataset.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    parser.add_argument('--project', type=str, default='D:/smartparking/runs/train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    return parser.parse_args()

def main(opt):
    # Run the training
    train.run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)