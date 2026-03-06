import os
import tempfile
import yt_dlp
import streamlit as st
import torchvision.transforms as transforms
from scripts.inference import load_models, extract_audio_features, generate_image_from_audio

# Set up page configurations
st.set_page_config(
    page_title="Audio-to-Abstract Art",
    page_icon="🎨",
    layout="centered"
)

st.title("Synesthesia: Audio to Abstract Art 🎶➡️🖼️")
st.markdown("Upload a music track or paste a YouTube link, and watch it turn into abstract art through completely offline AI!")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def get_models():
    """Cache models in memory so they only load once on startup."""
    return load_models(BASE_DIR)

# Load constraints & models
try:
    audio_vae, image_vae, mapping_net = get_models()
except Exception as e:
    st.error(f"Failed to load pre-trained models. Ensure `tmodels/` directory is present in the root. Error: {e}")
    st.stop()


def process_and_generate(audio_path):
    """Helper functional wrapper to run inference and render result"""
    st.audio(audio_path)
    
    with st.spinner("Extracting audio features (MFCCs, Tempo, Spectrum)..."):
        try:
            audio_features, condition = extract_audio_features(audio_path)
        except Exception as e:
            st.error(f"Error extracting features: {e}")
            return

    with st.spinner("Mapping latent space & generating art..."):
        try:
            generated_tensor = generate_image_from_audio(
                audio_features, condition, audio_vae, mapping_net, image_vae
            )
            
            # Convert PyTorch tensor (C, H, W) to PIL Image for Streamlit
            img_tensor = generated_tensor.squeeze(0).cpu()  # Remove batch dim
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(img_tensor)
            
            st.success("Art generated successfully!")
            st.image(pil_image, caption="Generated Abstract Art", use_container_width=True)
            
        except Exception as e:
            st.error(f"Inference mapping failed: {e}")

# Option Toggle
input_method = st.radio("Choose Input Method", ("Upload Audio File", "YouTube Link"))

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if uploaded_file is not None:
        # Save uploaded file to temp file for librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        process_and_generate(tmp_path)
        
        # Cleanup temp file
        os.remove(tmp_path)

elif input_method == "YouTube Link":
    yt_url = st.text_input("Enter YouTube URL (e.g., https://www.youtube.com/watch?v=...)")
    if yt_url:
        st.info("Downloading audio from YouTube... Please wait.")
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                    'preferredquality': '192',
                }],
                'outtmpl': os.path.join(temp_dir, 'yt_audio.%(ext)s'),
                'quiet': True,
                'no_warnings': True
            }
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(yt_url, download=True)
                    audio_filename = ydl.prepare_filename(info_dict)
                    base, ext = os.path.splitext(audio_filename)
                    final_audio_path = base + ".mp3"
                
                process_and_generate(final_audio_path)
            
            except Exception as e:
                st.error(f"Failed to fetch YouTube audio. Make sure the link is valid. Error: {e}")
