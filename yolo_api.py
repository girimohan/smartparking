import os
import sys
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Add the YOLOv5 directory to the Python path
yolov5_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'yolov5'))
sys.path.append(yolov5_path)

from yolov5.parking_detection import ParkingDetector

app = Flask(__name__)
detector = ParkingDetector()

# Define the upload folder
UPLOAD_FOLDER = os.path.normpath('D:/smartparking/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create directory if it doesn't exist

@app.route('/check_parking', methods=['POST'])
def check_parking():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Process image
        available_spaces, detected_image_path = detector.detect_parking_spaces(file_path)
        return jsonify({"available_spaces": available_spaces, "detected_image_path": detected_image_path})
    elif filename.lower().endswith(('.mp4', '.avi', '.mov')):
        # Process video
        processed_video_path, average_available_spaces = process_video(file_path)
        return jsonify({
            "message": "Video processed",
            "processed_video_path": processed_video_path,
            "average_available_spaces": average_available_spaces
        })
    else:
        return jsonify({"error": "Unsupported file type"}), 400

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None, 0
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = f'processed_{base_name}.mp4'
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Video properties - Width: {width}, Height: {height}, FPS: {fps}")
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_frames = 0
    total_available_spaces = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 5 != 0:
            continue

        img = detector.preprocess_image_from_frame(frame)
        with torch.no_grad():
            pred = detector.model(img)[0]
        pred = detector.non_max_suppression(pred, 0.25, 0.45)
        
        available_spaces = sum(1 for det in pred[0] if det[-1] == 0)
        total_available_spaces += available_spaces
        total_frames += 1

        frame_with_boxes = detector.draw_boxes_on_frame(frame, pred)
        cv2.putText(frame_with_boxes, f"Available Spaces: {available_spaces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame_with_boxes)

    cap.release()
    out.release()

    average_available_spaces = total_available_spaces // total_frames if total_frames > 0 else 0
    print(f"Processed video saved at: {os.path.normpath(output_path)}")
    print(f"Average available spaces: {average_available_spaces}")

    return os.path.normpath(output_path), average_available_spaces

@app.route('/check_parking_image', methods=['GET'])
def check_parking_image():
    image_path = os.path.join(UPLOAD_FOLDER, 'latest_parking_image.jpg')
    
    if not os.path.exists(image_path):
        return jsonify({"error": "No current parking image available"}), 404

    img = detector.preprocess_image(image_path)
    with torch.no_grad():
        pred = detector.model(img)[0]
    pred = detector.non_max_suppression(pred, 0.25, 0.45)
    
    available_spaces = sum(1 for det in pred[0] if det[-1] == 0)
    
    return jsonify({"available_spaces": available_spaces})

if __name__ == '__main__':
    app.run(debug=False, port=5000) #debug is set as false to avoid error while running all services combinely
