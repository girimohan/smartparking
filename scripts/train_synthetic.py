import os
import sys
from pathlib import Path
import torch



# Path to your YOLOv5 directory
yolov5_dir = Path('D:/smartparking/yolov5')

# Ensure YOLOv5 is in the Python path
sys.path.append(str(yolov5_dir))

# Dataset and model paths
dataset_yaml = Path('D:/smartparking/synthetic_pklot_dataset/dataset.yaml')
pretrained_weights = Path('D:/smartparking/runs/train/best.pt')

# Train the YOLOv5 model using the train.py script from YOLOv5
def train_synthetic():
    command = f'python {yolov5_dir}/train.py --img 640 --batch 8 --epochs 10 --data {dataset_yaml} --weights {pretrained_weights} --name synthetic_data_finetune'
    print(f'Executing command: {command}')  # Add this line for debugging
    os.system(command)

if __name__ == "__main__":
    train_synthetic()
