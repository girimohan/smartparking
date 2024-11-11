import sys
import os
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import argparse
import yaml

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Now import YOLOv5 modules
from yolov5.models.yolo import Model
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import check_img_size, labels_to_class_weights, init_seeds
from yolov5.utils.torch_utils import select_device
from yolov5.utils.loss import ComputeLoss
from yolov5.utils.plots import plot_results
from yolov5.utils.metrics import fitness

def train(hyp, opt, device):
    save_dir = Path(opt.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)
    train_path = os.path.join(data_dict['path'], data_dict['train'])
    val_path = os.path.join(data_dict['path'], data_dict['val'])
    nc = int(data_dict['nc'])
    
    # Create model
    model = Model(opt.cfg, ch=3, nc=nc).to(device)
    
    # Optimizer
    nbs = 64
    accumulate = max(round(nbs / opt.batch_size), 1)
    hyp['weight_decay'] *= opt.batch_size * accumulate / nbs
    optimizer = optim.Adam(model.parameters(), lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))
    
    # Scheduler
    lf = lambda x: (1 - x / opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # Dataloader
    imgsz = check_img_size(opt.img_size, s=32)  # Ensure image size is divisible by 32
    train_loader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, 32, opt,
                                              hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect)
    
    # Start training
    model.nc = nc
    model.hyp = hyp
    model.gr = 1.0
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)
    
    # Compute loss
    compute_loss = ComputeLoss(model)
    
    print(f"Starting training for {opt.epochs} epochs...")
    for epoch in range(opt.epochs):
        model.train()
        mloss = torch.zeros(1, device=device)  # mean losses
        pbar = enumerate(train_loader)
        print(f"Epoch {epoch+1}/{opt.epochs}")
        for i, (imgs, targets, paths, _) in pbar:
            imgs = imgs.to(device, non_blocking=True).float() / 255.0
            targets = targets.to(device)
            
            # Forward
            pred = model(imgs)
            
            # Compute loss
            loss, loss_items = compute_loss(pred, targets)
            
            # Backward
            loss.backward()
            
            # Optimize
            if i % accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Print
            mloss = (mloss * i + loss) / (i + 1)
            if i % 100 == 0:
                print(f"Epoch {epoch+1}/{opt.epochs}, Batch {i}/{len(train_loader)}, Loss: {mloss.item():.4f}")
        
        # Update scheduler
        scheduler.step()
        
        # Save model
        if (epoch + 1) % opt.save_interval == 0:
            ckpt = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(ckpt, save_dir / f'epoch_{epoch + 1}.pt')
            print(f"Saved checkpoint at epoch {epoch + 1}")
    
    # Final save
    torch.save(model.state_dict(), save_dir / 'final.pt')
    print("Training completed. Final model saved.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default=str(parent_dir / 'yolov5' / 'models' / 'yolov5s.yaml'), help='model.yaml path')
    parser.add_argument('--data', type=str, default='D:/smartparking/pklot_dataset/pklot_dataset.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default=str(parent_dir / 'yolov5' / 'data' / 'hyps' / 'hyp.scratch-low.yaml'), help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', default='D:/smartparking/runs/train', help='save to project/name')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--save-interval', type=int, default=10, help='Save checkpoint every x epochs')
    opt = parser.parse_args()
    
    # Set device
    device = select_device(opt.device)
    
    # Load hyperparameters
    hyp_file = Path(opt.hyp)
    if not hyp_file.exists():
        print(f"Hyperparameters file not found: {opt.hyp}")
        print("Please make sure the file exists and the path is correct.")
        return

    try:
        with open(opt.hyp) as f:
            hyp = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Hyperparameters file not found: {opt.hyp}")
        print("Please make sure the file exists and the path is correct.")
        return

    print("Configuration:")
    print(f"Image size: {opt.img_size}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Epochs: {opt.epochs}")
    print(f"Device: {device}")

    # Train
    init_seeds(1)
    train(hyp, opt, device)

if __name__ == '__main__':
    main()