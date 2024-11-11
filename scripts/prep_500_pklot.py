import os
import shutil
import json
import random
from tqdm import tqdm
from collections import Counter

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x, y, width, height = bbox
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

def coco_to_yolo_class(coco_category_id):
    if coco_category_id == 1:
        return 0  # "space-empty"
    elif coco_category_id == 2:
        return 1  # "space-occupied"
    else:
        raise ValueError(f"Unexpected category ID: {coco_category_id}")

def process_annotations(annotation_path, src_split, dst_split, num_images):
    src_img_dir = f'D:/PKLot_dataset/{src_split}/'
    dst_img_dir = f'D:/smartparking/pklot_1500/images/{dst_split}/'
    os.makedirs(dst_img_dir, exist_ok=True)
    
    label_dir = f'D:/smartparking/pklot_1500/labels/{dst_split}/'
    os.makedirs(label_dir, exist_ok=True)
    
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    selected_images = random.sample(data['images'], min(num_images, len(data['images'])))
    class_counter = Counter()

    for item in tqdm(selected_images, desc=f"Processing {src_split} annotations"):
        img_filename = item['file_name']
        src_img_path = os.path.join(src_img_dir, img_filename)
        dst_img_path = os.path.join(dst_img_dir, img_filename)
        
        try:
            shutil.copy(src_img_path, dst_img_path)
        except FileNotFoundError:
            print(f"File not found: {src_img_path}")
            continue
        
        img_id = item['id']
        labels = []
        
        for annotation in data['annotations']:
            if annotation['image_id'] == img_id:
                x_center, y_center, width, height = coco_to_yolo_bbox(annotation['bbox'], item['width'], item['height'])
                yolo_class_id = coco_to_yolo_class(annotation['category_id'])
                labels.append(f"{yolo_class_id} {x_center} {y_center} {width} {height}")
                class_counter[yolo_class_id] += 1
        
        label_file = os.path.join(label_dir, img_filename.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"Finished processing {len(selected_images)} images from {src_split} and saved to {dst_split} directory.")
    print(f"Class distribution in {dst_split}:")
    for class_id, count in class_counter.items():
        print(f"Class {class_id} ({'space-empty' if class_id == 0 else 'space-occupied'}): {count} instances")

def main():
    train_annotations = 'D:/PKLot_dataset/train/_annotations.coco.json'
    test_annotations = 'D:/PKLot_dataset/test/_annotations.coco.json'
    valid_annotations = 'D:/PKLot_dataset/valid/_annotations.coco.json'

    num_train = 1050
    num_test = 225
    num_val = 225

    process_annotations(train_annotations, 'train', 'train', num_train)
    process_annotations(test_annotations, 'test', 'test', num_test)
    process_annotations(valid_annotations, 'valid', 'val', num_val)

    # Create YAML file
    yaml_content = f"""
    path: D:/smartparking/pklot_1500
    train: images/train
    val: images/val
    test: images/test

    nc: 2
    names: ['space-empty', 'space-occupied']
    """
    
    with open('D:/smartparking/pklot_1500/pklot_1500.yaml', 'w') as f:
        f.write(yaml_content.strip())
    print("Created pklot_1500.yaml file.")

if __name__ == "__main__":
    main()