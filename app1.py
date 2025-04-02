import json
import cv2
import os
from PIL import Image
import requests
from io import BytesIO
import streamlit as st
import google.generativeai as gen_ai

# Set Streamlit Page Configuration
st.set_page_config(page_title="Image & Video Caption Generator", page_icon="📷", layout="centered")

# Load API Key from config.json
try:
    with open("config.json") as config_file:
        config_data = json.load(config_file)
        GOOGLE_API_KEY = config_data.get("GOOGLE_API_KEY", "")
except (FileNotFoundError, json.JSONDecodeError):
    st.error("Error loading API key from config.json. Make sure the file exists and is correctly formatted.")
    st.stop()

# Configure Google Generative AI
if GOOGLE_API_KEY:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
else:
    st.error("Missing Google API Key! Please check config.json.")
    st.stop()

# Load Gemini Model
model = gen_ai.GenerativeModel('gemini-1.5-flash')

def get_image_caption(prompt, image, model=model):
    try:
        response = model.generate_content([prompt, image])
        return response.text if response.text else "No caption generated."
    except Exception as e:
        return f"Error generating caption: {str(e)}"

def extract_frames(video_path, output_folder, frame_rate=30):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, image = cap.read()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frames = []
    while success:
        if frame_count % frame_rate == 0:  # Capture every 'frame_rate' frames
            frame_filename = os.path.join(output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, image)
            frames.append(frame_filename)
        success, image = cap.read()
        frame_count += 1
    
    cap.release()
    return frames

def generate_video_captions(video_path, model):
    frames_folder = "video_frames"
    frames = extract_frames(video_path, frames_folder)
    
    captions = []
    for frame in frames:
        image = Image.open(frame)
        caption = get_image_caption("Describe this scene", image, model)
        captions.append(caption)
    
    return captions

def summarize_captions(captions):
    prompt = "Summarize these video scene descriptions into a meaningful caption:\n" + "\n".join(captions)
    response = model.generate_content(prompt)
    return response.text if response.text else "No summary generated."

st.title("📷 Image & Video Caption Generator")

# Select Source Type
source_type = st.radio("Select Source Type:", ("Upload Image", "Enter Image URL", "Upload Video"))

def process_image(image):
    resized_img = image.resize((800, 500))
    st.image(resized_img)
    caption = get_image_caption("Write a short caption for this image.", image)
    st.info(caption)

if source_type == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None and st.button("Generate Caption"):
        image = Image.open(uploaded_image)
        process_image(image)

elif source_type == "Enter Image URL":
    image_url = st.text_input("Enter Image URL:")
    if st.button("Generate Caption"):
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content))
            process_image(image)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

elif source_type == "Upload Video":
    video_file = st.file_uploader("Upload a video...", type=["mp4", "avi", "mov"])
    if video_file is not None and st.button("Generate Video Caption"):
        with open("uploaded_video.mp4", "wb") as f:
            f.write(video_file.read())
        
        captions = generate_video_captions("uploaded_video.mp4", model)
        summary = summarize_captions(captions)
        
        st.info(summary)
