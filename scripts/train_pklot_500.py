import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # This points to the parent of the 'scripts' directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# Now we can import from yolov5
from yolov5 import train

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--data', type=str, default='D:/smartparking/pklot_1500/pklot_1500.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--project', type=str, default=ROOT / 'runs' / 'train', help='save to project/name')
    parser.add_argument('--name', type=str, default='exp', help='save to project/name')
    return parser.parse_args()

def main(opt):
    train.run(**vars(opt))

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)