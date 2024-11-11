import numpy as np
import cv2
import os
from random import randint, choice

def create_parking_lot(width, height, rows, spots_per_row):
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    spot_width = width // spots_per_row
    spot_height = height // rows
    annotations = []

    for row in range(rows):
        for spot in range(spots_per_row):
            x1 = spot * spot_width
            y1 = row * spot_height
            x2 = x1 + spot_width
            y2 = y1 + spot_height
            
            # Draw parking spot
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)
            
            # Randomly decide if the spot is occupied
            is_occupied = randint(0, 1)
            class_id = 1 if is_occupied else 0  # 1 for occupied, 0 for vacant
            
            if is_occupied:
                car_color = (randint(100, 255), randint(100, 255), randint(100, 255))
                cv2.rectangle(image, (x1+2, y1+2), (x2-2, y2-2), car_color, -1)
            
            # Generate YOLO format annotation for every spot
            center_x = (x1 + x2) / (2 * width)
            center_y = (y1 + y2) / (2 * height)
            box_width = (x2 - x1) / width
            box_height = (y2 - y1) / height
            annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")

    return image, annotations

def generate_dataset(num_images, base_dir):
    splits = {
        'train': int(num_images * 0.7),
        'val': int(num_images * 0.15),
        'test': int(num_images * 0.15)
    }

    for split, count in splits.items():
        img_dir = os.path.join(base_dir, 'images', split)
        label_dir = os.path.join(base_dir, 'labels', split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        for i in range(count):
            width, height = 640, 480
            rows = randint(3, 5)
            spots_per_row = randint(8, 12)
            
            image, annotations = create_parking_lot(width, height, rows, spots_per_row)
            
            # Save image
            img_path = os.path.join(img_dir, f'parking_lot_{split}_{i:04d}.jpg')
            cv2.imwrite(img_path, image)
            
            # Save annotations
            label_path = os.path.join(label_dir, f'parking_lot_{split}_{i:04d}.txt')
            with open(label_path, 'w') as f:
                f.write('\n'.join(annotations))

        print(f"Generated {count} images and annotations for {split} set")

# Generate the dataset
output_directory = 'D:/smartparking/synthetic_pklot_dataset'
generate_dataset(1500, output_directory)

# Create dataset.yaml file
yaml_content = f"""
path: {output_directory}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['vacant', 'occupied']
"""

with open(os.path.join(output_directory, 'dataset.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"Dataset YAML file created at {os.path.join(output_directory, 'synthetic_dataset.yaml')}")
