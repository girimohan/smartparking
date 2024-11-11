import os
import shutil
import json
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_width, img_height):
    x, y, width, height = bbox
    x_center = (x + width / 2) / img_width
    y_center = (y + height / 2) / img_height
    width /= img_width
    height /= img_height
    return x_center, y_center, width, height

def process_annotations(annotation_path, src_split, dst_split):
    # Define source and destination paths
    src_img_dir = f'D:/PKLot_dataset/{src_split}/'
    dst_img_dir = f'D:/smartparking/pklot_dataset/images/{dst_split}/'
    os.makedirs(dst_img_dir, exist_ok=True)
    
    label_dir = f'D:/smartparking/pklot_dataset/labels/{dst_split}/'
    os.makedirs(label_dir, exist_ok=True)
    
    # Open and parse the annotation file
    with open(annotation_path, 'r') as f:
        data = json.load(f)

    for item in tqdm(data['images'], desc=f"Processing {src_split} annotations"):
        img_filename = item['file_name']
        src_img_path = os.path.join(src_img_dir, img_filename)
        dst_img_path = os.path.join(dst_img_dir, img_filename)
        
        # Copy the image file, handle missing files
        try:
            shutil.copy(src_img_path, dst_img_path)
        except FileNotFoundError:
            print(f"File not found: {src_img_path}")
            continue
        
        # Generate label file
        img_id = item['id']
        labels = []
        
        for annotation in data['annotations']:
            if annotation['image_id'] == img_id:
                x_center, y_center, width, height = coco_to_yolo_bbox(annotation['bbox'], item['width'], item['height'])
                class_id = annotation['category_id'] - 1  # Adjust class_id if necessary
                labels.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        label_file = os.path.join(label_dir, img_filename.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            f.write('\n'.join(labels))
    
    print(f"Finished processing {src_split} annotations and saved to {dst_split} directory.")

def main():
    # Paths to annotation files
    train_annotations = 'D:/PKLot_dataset/train/_annotations.coco.json'
    test_annotations = 'D:/PKLot_dataset/test/_annotations.coco.json'
    valid_annotations = 'D:/PKLot_dataset/valid/_annotations.coco.json'

    # Process each split
    process_annotations(train_annotations, 'train', 'train')
    process_annotations(test_annotations, 'test', 'test')
    process_annotations(valid_annotations, 'valid', 'val')  # Source is 'valid', destination is 'val'

if __name__ == "__main__":
    main()