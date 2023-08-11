import os
from pathlib import Path
import base64
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import IPython
from IPython.display import HTML, display


def stereo_to_mono(data):
    """Convert stereo audio to mono by averaging the samples"""
    mono_audio = np.mean(data, axis=1, dtype=data.dtype)
    return mono_audio


def resample(data, orig_sr, target_sr):
    """resample down/up a data audio"""
    ratio = orig_sr / target_sr
    nums = int(len(data) / ratio)
    if len(data.shape) > 1:
        data = stereo_to_mono(data)
    sampled = signal.resample(data, nums)
    return sampled


def table_with_sound(dframe, audio_column='audio_column', columns=[], pred=False):
    """
        Helper function for show dataframe with audio-play column
    """

    def path_to_sound_html(sound):
        encode_string = base64.b64encode(open(sound[audio_column], "rb").read())
        encode_string = encode_string.decode("utf-8")
        encode_string = 'data:audio/x-wav;base64,' + encode_string
        return f'<audio controls ><source src={encode_string} type="audio/x-wav" />Your browser sucks!</audio> '

    dframe[audio_column] = dframe.apply(path_to_sound_html, axis=1)

    _columns = columns
    if pred:
        _columns.append('Prediction')

    return HTML(dframe[_columns].to_html(escape=False))
