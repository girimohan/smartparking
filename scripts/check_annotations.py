import cv2
import os

# Paths to your dataset
images_path = 'D:/smartparking/pklot_1500/images/train/'
labels_path = 'D:/smartparking/pklot_1500/labels/train/'

def draw_yolo_annotations(image_path, label_path):
    # Load image
    image = cv2.imread(image_path)
    h, w, _ = image.shape
    
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1]) * w
            y_center = float(parts[2]) * h
            width = float(parts[3]) * w
            height = float(parts[4]) * h
            
            # Calculate bounding box coordinates
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Set color based on class
            color = (0, 255, 0)  # Default color (green)
            if class_id == 0:
                color = (255, 0, 0)  # Red for space-empty
            elif class_id == 1:
                color = (0, 255, 0)  # Green for space-occupied
            
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f'Class {class_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    return image


def main():
    for image_file in os.listdir(images_path):
        if image_file.endswith('.jpg'):  # or '.png', depending on your dataset
            image_id = os.path.splitext(image_file)[0]
            label_file = os.path.join(labels_path, f'{image_id}.txt')
            
            if os.path.exists(label_file):
                image_path = os.path.join(images_path, image_file)
                annotated_image = draw_yolo_annotations(image_path, label_file)
                
                # Display the image
                cv2.imshow('Annotated Image', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
