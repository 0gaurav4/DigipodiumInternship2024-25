import streamlit as st
import os
import google.generativeai as genai
from io import BytesIO
from PIL import Image
from gtts import gTTS
from langdetect import detect
from deep_translator import GoogleTranslator
import tempfile

# Configure Gemini API
genai.configure(api_key="AIzaSyBElhgGXpSEQ4x-yO7-BUZXBNHeSfzidEE")  # Replace with your actual API key

# --- Helper Functions ---
def upload_to_gemini(image_file, mime_type=None):
    if image_file:
        try:
            image = Image.open(image_file)
            image_bytes = BytesIO()
            image.save(image_bytes, format='JPEG')
            image_bytes.seek(0)
            file = genai.upload_file(image_bytes, mime_type=mime_type)
            st.session_state.uploaded_file = file
            return file
        except Exception as e:
            st.error(f"Failed to upload the file to Gemini: {e}")
            return None
    return None

def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        st.error(f"Error detecting language: {e}")
        return None

def translate_text(text, dest_language):
    try:
        return GoogleTranslator(source='auto', target=dest_language).translate(text)
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return None

def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# --- Streamlit App ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_stage" not in st.session_state:
    st.session_state.current_stage = 0
if "selected_language" not in st.session_state:
    st.session_state.selected_language = None
if "detected_language" not in st.session_state:
    st.session_state.detected_language = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
    system_instruction=(
        "-ask user to upload image\n"
        "-detect text from images and language of the text\n"
        "-translate text to selected language\n"
        "strictly give only main translated text from photo and give no other text\n"
    ),
)

st.title("Indian Language Image to Speech Translator")

# Define the available languages
indian_languages = {
    "Hindi": "hi",
    "English": "en",
    "Marathi": "mr",
    "Telugu": "te",
    "Tamil": "ta",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Odia": "or",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# Select the language first
st.session_state.selected_language = st.selectbox("Select the language for translation and audio:", list(indian_languages.keys()))

if st.session_state.selected_language:
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        try:
            st.session_state.uploaded_image = uploaded_file
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            gemini_file = upload_to_gemini(uploaded_file, mime_type="image/jpeg")
            if gemini_file:
                st.session_state.chat_history = [{"role": "user", "parts": [gemini_file]}]
                with st.spinner("Detecting text and language..."):
                    chat_session = model.start_chat(history=st.session_state.chat_history)
                    response = chat_session.send_message("Detect text and language of text from image")
                    st.session_state.chat_history.append({"role": "model", "parts": [response.text]})

                    detected_text = response.text.strip()
                    st.session_state.detected_language = detect_language(detected_text)
                    st.success(f"Detected Language: {st.session_state.detected_language}")
                    

                    with st.spinner("Translating text..."):
                        response = chat_session.send_message(f"translate the text to {st.session_state.selected_language}")
                        st.session_state.chat_history.append({"role": "model", "parts": [response.text]})

                        translated_text = response.text.strip()
                        st.write("### Translated Text:")
                        st.success(translated_text)

                        with st.spinner("Generating audio..."):
                            target_language_code = indian_languages[st.session_state.selected_language]
                            audio_file = text_to_speech(translated_text, target_language_code)
                            if audio_file:
                                st.audio(audio_file, format="audio/mp3", start_time=0, autoplay=True)
                                with open(audio_file, "rb") as f:
                                    audio_bytes = f.read()
                                st.download_button(
                                    label="Download Audio",
                                    data=audio_bytes,
                                    file_name="translated_audio.mp3",
                                    mime="audio/mp3"
                                )
                                os.remove(audio_file)
        except Exception as e:
            st.error(f"Error processing the image: {e}")

if st.button("Restart"):
    st.session_state.current_stage = 0
    st.session_state.chat_history = []
    st.session_state.selected_language = None
    st.session_state.detected_language = None
    st.session_state.uploaded_image = None
    st.rerun()
