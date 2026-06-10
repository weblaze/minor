import io
import os
import tempfile

import streamlit as st
import torchvision.transforms as transforms
import yt_dlp

from abstraction.pipelines.latent_diffusion import generate_diffusion, load_models
from abstraction.utils.config import load_config

st.set_page_config(
    page_title="Abstraction: Music-to-Image AI",
    page_icon="🎨",
    layout="centered"
)

st.title("Abstraction: Music to Abstract Art 🎶➡️🖼️")
st.markdown("""
**Abstraction** generates abstract visual artwork directly from audio timbre and rhythm.
It uses no text prompts—only the raw musical waveform.
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "approaches", "02_latent_diffusion_clap", "config.yaml")


@st.cache_resource
def get_models():
    """Cache config + models in memory so they only load once on startup."""
    config = load_config(CONFIG_PATH)
    return config, load_models(config)


try:
    config, (clap_model, unet, codec, scheduler) = get_models()
except Exception as e:
    st.error(f"Failed to load Latent Diffusion models. Error: {e}")
    st.stop()


def process_and_generate(audio_path=None, is_dreaming=False):
    """Run diffusion inference and render the result."""
    if not is_dreaming:
        st.audio(audio_path)
        status_text = "Analyzing musical timbre & rhythm..."
    else:
        status_text = "Accessing internal musical imagination..."

    progress_bar = st.progress(0)
    progress_text = st.empty()

    def update_progress(pct):
        progress_bar.progress(pct)
        progress_text.text(f"Denoising Latent Space: {pct}%")

    with st.spinner(status_text):
        try:
            generated_tensor = generate_diffusion(
                clap_model, unet, codec, scheduler,
                audio_path=audio_path if not is_dreaming else None,
                num_steps=config["sampling"]["num_steps"],
                guidance_scale=st.session_state.get("guidance_scale", config["sampling"]["guidance_scale"]),
                progress_callback=update_progress
            )

            pil_image = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())

            st.success("Art generated successfully!")
            st.image(pil_image, caption="AI Dream Output" if is_dreaming else "Generated Abstract Art",
                     use_column_width=True)

            buf = io.BytesIO()
            pil_image.save(buf, format="PNG")
            st.download_button(
                label="Download Artwork",
                data=buf.getvalue(),
                file_name="abstraction_output.png",
                mime="image/png"
            )
        except Exception as e:
            st.error(f"Inference failed: {e}")


with st.sidebar:
    st.header("Settings")
    st.slider(
        "Guidance scale", min_value=1.0, max_value=10.0,
        value=float(config["sampling"]["guidance_scale"]), step=0.5,
        key="guidance_scale",
        help="How strongly the music steers the image. 1 = unguided, higher = more literal."
    )

    st.divider()
    st.subheader("Experimental: Dreaming")
    st.write("Generate imagery from the model's internal statistical distribution without any audio input.")
    if st.button("✨ Start Dreaming"):
        process_and_generate(is_dreaming=True)

input_method = st.radio("Choose Input Method", ("Upload Audio File", "YouTube Link"))

if input_method == "Upload Audio File":
    uploaded_file = st.file_uploader("Upload an MP3 or WAV file", type=["mp3", "wav"])
    if uploaded_file is not None:
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
