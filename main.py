import json
from PIL import Image # Python Imaging Library
import requests # used for HTTP server
from io import BytesIO # used for Converting downloaded data (like images) into a format readable by PIL or OpenCV.
import streamlit as st # used for web application 
import google.generativeai as gen_ai #  Integrates Google's generative AI models (like Gemini) for text/image processing.

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="📷",
    layout="centered",
)

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

# Load Gemini 1.5 Flash Model
model = gen_ai.GenerativeModel('gemini-1.5-flash')

def get_image_caption(prompt, image, model=model):
    try:
        response = model.generate_content([prompt, image])
        return response.text if response.text else "No caption generated."
    except Exception as e:
        return f"Error generating caption: {str(e)}"

st.title("📷 Image Caption Generator ")

# Select Image Source
image_source = st.radio("Select Image Source:", ("Upload an image", "Enter Image URL"))

def process_image(image):
    resized_img = image.resize((800, 500))
    st.image(resized_img)
    default_prompt = "Write a short caption for this image."
    caption = get_image_caption(default_prompt, image)
    st.info(caption)

if image_source == "Upload an image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None and st.button("Generate Caption"):
        image = Image.open(uploaded_image)
        process_image(image)
else:
    image_url = st.text_input("Enter Image URL:")
    if st.button("Generate Caption"):
        if image_url:
            try:
                # Validate the URL
                response = requests.get(image_url, timeout=5)
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    
                    # Check if URL is an image
                    if "image" in content_type:
                        image = Image.open(BytesIO(response.content))
                        process_image(image)
                    else:
                        st.error("The provided URL does not contain a valid image.")
                else:
                    st.error(f"Failed to fetch image, HTTP status code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching image: {str(e)}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
