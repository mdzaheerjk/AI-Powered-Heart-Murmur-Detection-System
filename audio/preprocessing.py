import librosa
import numpy as np
from config import SAMPLE_RATE,N_MFCC
from utils.logger import setup_logger

logger=setup_logger("AudioPreprocessing")

def load_audio(upload_file):
    try:
        logger.info("Loading audio file")
        y,sr=librosa.load(upload_file,sr=SAMPLE_RATE)
        return y,sr
    except Exception as e:
        logger.exception("Audio Loading failed")
        raise RuntimeError("Invalid or currupted audio file") from e

def extract_mfcc(y,sr):
    try:
        logger.info("Extracting MFCC Features")
        mfcc=librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=N_MFCC
        )
        mfcc_scaled=np.mean(mfcc.T,axis=0)
        x_input=np.expand_dims(mfcc_scaled,axis=0)
        x_input=np.expand_dims(x_input,axis=2)

        return x_input
    except Exception as e:
        logger.exception("MFCC extraction failed")
        raise RuntimeError('Feature extraction Failed') from e