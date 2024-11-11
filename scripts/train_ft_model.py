

import os
import sys
from pathlib import Path
import yaml

# Path to your YOLOv5 directory
yolov5_dir = Path('D:/smartparking/yolov5')
sys.path.append(str(yolov5_dir))

# Dataset and model paths
pklot_1500_yaml = 'D:/smartparking/pklot_1500/pklot_1500.yaml'
synthetic_finetuned_weights = '../yolov5/runs/train/synthetic_finetune/weights/best.pt'

# Path to the hyperparameters file
hyp_path = Path(yolov5_dir) / 'data' / 'hyps' / 'hyp.scratch-low.yaml'

def update_hyperparameters(hyp_path, lr=0.001):
    with open(hyp_path, 'r') as f:
        hyp = yaml.safe_load(f)
    
    hyp['lr0'] = lr  # Set initial learning rate
    hyp['lrf'] = lr / 10  # Set final learning rate

    with open(hyp_path, 'w') as f:
        yaml.dump(hyp, f)

def train_on_pklot_1500():
    # Update hyperparameters
    update_hyperparameters(hyp_path)

    command = f"""
    python {yolov5_dir}/train.py 
    --img 640 
    --batch 8 
    --epochs 10
    --data {pklot_1500_yaml} 
    --weights {synthetic_finetuned_weights} 
    --name pklot_1500_after_synthetic 
    --freeze 10 
    --save-period 5
    --hyp {hyp_path}
    """
    os.system(command.replace('\n', ' '))

if __name__ == "__main__":
    train_on_pklot_1500()