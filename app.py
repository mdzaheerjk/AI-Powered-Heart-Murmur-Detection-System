import streamlit as st
import numpy as np

from model.model_loader import load_model
from audio.preprocessing import load_audio, extract_mfcc
from ui.visualizations import plot_waveform
from utils.logger import setup_logger

logger = setup_logger("Streamlit App")

st.set_page_config(
    page_title="Heart Murmur Detection",
    page_icon="❤️",
    layout="centered"
)


model = load_model()

class_names = {
    0: "artifacts",
    1: "murmur",
    2: "normal"
}

st.title("❤️ Heart Murmur Detection with LSTM")

st.write(
    "Upload a heart sound and click **Predict** "
    "to classify it as Artifacts, Murmur, or Normal."
)


uploaded_file = st.file_uploader(
    "Upload a heart sound (WAV/MP3)",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    st.subheader("📁 Uploaded Audio")

    st.write(f"**File:** {uploaded_file.name}")

    # Audio player
    st.subheader("🔊 Listen to Audio")

    st.audio(uploaded_file)

    if st.button(
        "🔮 Predict",
        type="primary",
        use_container_width=True
    ):

        try:

            with st.spinner("Loading audio..."):

                y, sr = load_audio(uploaded_file)

            st.subheader("📈 Waveform")

            fig = plot_waveform(y, sr)

            st.pyplot(fig)

            with st.spinner("Extracting audio features..."):

                x_input = extract_mfcc(y, sr)

            with st.spinner("Running LSTM prediction..."):

                prediction = model.predict(
                    x_input,
                    verbose=0
                )


            predicted_class = np.argmax(
                prediction,
                axis=1
            )[0]

            predicted_name = class_names[
                predicted_class
            ]


            confidence = (
                np.max(prediction) * 100
            )

            st.subheader("🔮 Prediction Result")

            st.success(
                f"Predicted Class: "
                f"**{predicted_name.upper()}**"
            )

            st.info(
                f"Confidence: **{confidence:.2f}%**"
            )

            st.subheader("📊 Prediction Probabilities")

            for class_index, class_name in class_names.items():

                probability = (
                    prediction[0][class_index] * 100
                )

                st.write(
                    f"**{class_name.capitalize()}**: "
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(prediction[0][class_index])
                )


        except Exception as e:

            logger.exception(
                "Inference Pipeline Failed"
            )

            st.error(
                "⚠️ An error occurred while "
                "processing the audio file."
            )

            st.exception(e)