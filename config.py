import os

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

SAMPLE_RATE=22050
N_MFCC=52

HF_REPO_ID='zaheerjk/AI-Powered-Heart-Murmur-Detection-System'
HF_MODEL_FILENAME='Models/lstm_model.h5'