import matplotlib.pyplot as plt
import librosa.display
from utils.logger import setup_logger

logger = setup_logger("Visualization")

def plot_waveform(y, sr):
    try:
        logger.info("Plotting waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title("Waveform")
        return fig
    except Exception as e:
        logger.exception("Waveform plotting failed")
        raise RuntimeError("Failed to visualize waveform") from e
