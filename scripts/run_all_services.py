import subprocess
import os

# Paths to Rasa project, Flask API, and Streamlit project
rasa_project_path = "D:/smartparking/rasa_chatbot"  
flask_api_script = "D:/smartparking/yolo_api.py"  
streamlit_app_script = "D:/smartparking/app.py"  

# Start Rasa core server
subprocess.Popen(['rasa', 'run'], cwd=rasa_project_path)

# Start Rasa action server
subprocess.Popen(['rasa', 'run', 'actions'], cwd=rasa_project_path)

# Start Flask API server
subprocess.Popen(['python', flask_api_script])

# Start Streamlit app
subprocess.Popen(['streamlit', 'run', streamlit_app_script])

print("All services are running...")
