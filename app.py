import streamlit as st
import numpy as np

from model.model_loader import load_model
from audio.preprocessing import load_audio, extract_mfcc
from ui.visualizations import plot_waveform
from utils.logger import setup_logger

logger = setup_logger("StreamlitApp")

# -------------------------------
# Load Model
# -------------------------------
model = load_model()

st.title("🎵 Heart Murmur Detection with LSTM")

uploaded_file = st.file_uploader(
    "Upload a heart sound (WAV/MP3)",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    try:
        y, sr = load_audio(uploaded_file)

        st.subheader("Waveform of Input Sound")
        fig = plot_waveform(y, sr)
        st.pyplot(fig)

        X_input = extract_mfcc(y, sr)

        prediction = model.predict(X_input)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.subheader("🔮 Prediction Result")
        st.write(f"Predicted Class: **{predicted_class}**")
        st.write("Raw Prediction Scores:", prediction)

        logger.info("Prediction completed successfully")

    except Exception as e:
        logger.exception("Inference pipeline failed")
        st.error("⚠️ An error occurred while processing the audio file.")
