import os
import sys
from pathlib import Path
import pickle
import base64
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from src.futil import stereo_to_mono, resample


root_path = "/Users/beltre.wilton/Downloads/SER-Datasets/MSP-Conversation-1.1"
AUDIO_PATH = Path(f'{root_path}/Audio')

audios = []
default_sr = 16_000

# TODO: later - use this this to prepare torch Dataset based on 'data'
for a in AUDIO_PATH.rglob("*.wav"):
    sr, data = wavfile.read(a.__str__())
    if sr != default_sr:
        data = resample(data, sr, default_sr)
    if len(data.shape) > 1:
        data = stereo_to_mono(data)
    duration = data.shape[0] / sr
    audios.append((a.name, a.__str__(), duration))

with open('../data/audiolist.pkl', "wb") as pkl:
    pickle.dump(audios, pkl)
