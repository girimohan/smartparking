# import streamlit as st
# import requests
# from PIL import Image
# import io

# # Function to send messages to Rasa and get a response
# def get_response_from_rasa(message):
#     # Define the URL of the Rasa server
#     rasa_url = 'http://localhost:5005/webhooks/rest/webhook'
    
#     # Send a POST request with the user message
#     response = requests.post(rasa_url, json={"message": message})
    
#     # Check if the response was successful
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return [{"text": "Sorry, I couldn't reach the chatbot at the moment."}]

# # Define the Streamlit app
# def main():
#     st.title("Smart Parking Assistant")

#     # Add a file uploader to the Streamlit app
#     uploaded_file = st.file_uploader("Upload an image of a parking lot", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
#         # Prepare the file for sending to the Flask API
#         files = {'file': uploaded_file.getvalue()}
        
#         # Send the image to the Flask API
#         response = requests.post('http://localhost:5000/check_parking', files=files)
        
#         if response.status_code == 200:
#             data = response.json()
#             available_spaces = data.get('available_spaces', 0)
#             st.write(f"There are currently {available_spaces} parking spaces available.")
#         else:
#             st.write("Failed to get parking space information.")
    
#     # Chatbot interaction section
#     st.header("Chat with our Assistant")

#     # Create a text input field for the user to type their message
#     user_message = st.text_input("You:", "")

#     if st.button('Send'):
#         if user_message:
#             # Get the response from Rasa
#             response = get_response_from_rasa(user_message)
            
#             # Display the response
#             for r in response:
#                 st.text(f"Chatbot: {r['text']}")
#         else:
#             st.text("Please enter a message.")

# if __name__ == "__main__":
#     main()


# updated code for video uploading

import streamlit as st
import requests
import json

# Global variable to store parking availability
parking_data = {}

# Function to save parking data to a file
def save_parking_data(data):
    with open('parking_data.json', 'w') as f:
        json.dump(data, f)

# Function to load parking data from a file
def load_parking_data():
    try:
        with open('parking_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Function to get bot response
def get_bot_response(message):
    rasa_server_url = "http://localhost:5005/webhooks/rest/webhook"
    payload = {
        "sender": "user",
        "message": message
    }
    try:
        response = requests.post(rasa_server_url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()[0]['text'] if response.json() else "Sorry, I didn't understand that."
    except requests.RequestException as e:
        return f"Error communicating with the chatbot: {str(e)}"

def main():
    global parking_data
    parking_data = load_parking_data()
    
    st.title("Smart Parking Assistant")

    # File upload section
    uploaded_file = st.file_uploader("Upload an image or video clip", type=["jpg", "jpeg", "png", "mp4", "avi"])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        
        if file_type == 'image':
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        elif file_type == 'video':
            st.video(uploaded_file, format='video/mp4')

        files = {'file': (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post('http://localhost:5000/check_parking', files=files)

        if response.status_code == 200:
            data = response.json()
            if file_type == 'image':
                available_spaces = data.get('available_spaces', 0)
                st.success(f"{available_spaces} parking spaces are available.")
            else:  # video
                average_available_spaces = data.get('average_available_spaces', 0)
                st.success(f"Video processing complete. On average, {average_available_spaces} parking spaces are available.")
            
            location = "a_katu"  # You might want to make this dynamic in the future
            parking_data[location] = available_spaces if file_type == 'image' else average_available_spaces
            save_parking_data(parking_data)
        else:
            st.error(f"Failed to process {file_type}. Status code: {response.status_code}. Error: {response.text}")

    # Chatbot interface
    st.subheader("Chat with Parking Assistant")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("You:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        bot_response = get_bot_response(user_input)
        
        # Check if the bot is asking for a location and we have parking data
        if "What's your current location?" in bot_response and parking_data:
            bot_response += f"\nBased on our latest data:"
            for location, spaces in parking_data.items():
                bot_response += f"\n- {location}: {spaces} available spaces"
        
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)

if __name__ == "__main__":
    main()