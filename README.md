# Smart Parking Assistant

A real-time parking space detection system that combines computer vision with a conversational AI interface to help drivers find available parking spots efficiently.

## Overview

Smart Parking Assistant addresses the growing challenge of urban parking by leveraging:
* YOLOv5 object detection for real-time parking space monitoring
* Rasa-powered chatbot for natural language interaction
* Flask API for backend processing
* Streamlit for the user interface

## Key Features

* **Real-time Parking Detection**: Uses YOLOv5l model to analyze parking lot camera feeds and identify available spaces
* **Natural Language Interface**: Conversational AI chatbot helps users find parking through simple text queries
* **High Accuracy**: Achieves 97.9% precision in detecting parking spaces
* **Privacy-Focused**: Processes video locally and only stores aggregated parking data
* **User-Friendly Interface**: Simple web interface for video upload and parking queries

## Technical Implementation

### Components
* **Computer Vision**: YOLOv5l model trained on PKLot dataset and synthetic data
* **Chatbot**: Rasa framework for natural language understanding
* **Backend**: Flask API for handling video processing
* **Frontend**: Streamlit web interface

## Requirements

* Python 3.8+
* PyTorch 1.9.0
* OpenCV 4.5.3
* Flask 2.0.1
* Rasa 3.1
* Streamlit 1.0.0