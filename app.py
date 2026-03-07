import os
import tempfile
import yt_dlp
import streamlit as st
import torchvision.transforms as transforms
import torch
from scripts.inference import load_models, generate_diffusion

# Set up page configurations
st.set_page_config(
    page_title="Abstraction: Music-to-Image AI",
    page_icon="🎨",
    layout="centered"
)

st.title("Synesthesia: Music to Abstract Art 🎶➡️🖼️")
st.markdown("""
**Abstraction** generates abstract visual artwork directly from audio timbre and rhythm. 
It uses no text prompts—only the raw musical waveform.
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def get_models():
    """Cache models in memory so they only load once on startup."""
    return load_models(BASE_DIR)

# Load constraints & models
try:
    clap_model, unet, image_vae, scheduler = get_models()
except Exception as e:
    st.error(f"Failed to load Latent Diffusion models. Error: {e}")
    st.stop()

def process_and_generate(audio_path=None, is_dreaming=False):
    """Helper functional wrapper to run diffusion inference and render result"""
    
    if not is_dreaming:
        st.audio(audio_path)
        status_text = "Analyzing musical timbre & rhythm..."
    else:
        status_text = "Accessing internal musical imagination..."

    # Use a column layout for the progress 
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    def update_progress(pct):
        progress_bar.progress(pct)
        progress_text.text(f"Denoising Latent Space: {pct}%")

    with st.spinner(status_text):
        try:
            # Generate using Diffusion loop
            generated_tensor = generate_diffusion(
                clap_model, unet, image_vae, scheduler,
                audio_path=audio_path if not is_dreaming else None,
                num_steps=50, # Sufficient for DDPM/DDIM inference usually
                progress_callback=update_progress
            )
            
            # Convert PyTorch tensor (1, C, H, W) to PIL Image
            img_tensor = generated_tensor.squeeze(0).cpu() 
            to_pil = transforms.ToPILImage()
            pil_image = to_pil(img_tensor)
            
            st.success("Art generated successfully!")
            st.image(pil_image, caption="AI Dream Output" if is_dreaming else "Generated Abstract Art", use_column_width=True)
            
            # Download button for the generated art
            import io
            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button(
                label="Download Artwork",
                data=byte_im,
                file_name="abstraction_output.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"Inference failed: {e}")

# Sidebar for Dreaming Mode
with st.sidebar:
    st.header("Settings")
    st.info("Operating on GTX 1650 (4GB VRAM) optimized settings.")
    
    st.divider()
    st.subheader("Experimental: Dreaming")
    st.write("Generate imagery from the model's internal statistical distribution without any audio input.")
    if st.button("✨ Start Dreaming"):
        process_and_generate(is_dreaming=True)

# Main Option Toggle
input_method = st.radio("Choose Input Method", ("Upload Audio File", "YouTube Link"))

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if uploaded_file is not None:
        # Get extension from uploaded file name
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        process_and_generate(tmp_path)
        os.remove(tmp_path)

elif input_method == "YouTube Link":
    yt_url = st.text_input("Enter YouTube URL")
    if yt_url:
        st.info("Downloading audio... Please wait.")
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
                st.error(f"Failed to fetch YouTube audio: {e}")
