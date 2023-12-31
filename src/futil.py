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


def quantization(bits: int = 8):
    """Quantization sexample"""
    f16vector = np.array([1.2, -0.5, -4.3, 1.2, -3.1, 0.8, 2.4, 5.4], dtype=np.float16)
    int8_range = 2**(bits - 1) - 1
    absmax = np.max(f16vector)
    sfactor = int8_range / absmax
    q_vector = np.round(f16vector * sfactor).astype(np.int8)
    back_vector = q_vector / sfactor
    print(f'Simple {bits} Quantization \n')
    print('Input vector:')
    print(f16vector)
    print('- - - - - - - - - - - - - - - - - -- \n')
    print('Quantized vector:')
    print(q_vector)
    print('- - - - - - - - - - - - - - - - - -- \n')
    print('Recovered vector:')
    print(back_vector)
    print('- - - - - - - - - - - - - - - - - -- \n')


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

        inpus:
            -dframe: Datarow con audio name, audio path y duration
            -audio_column: Columna donde se encuentra el audio
            -columns: Nombre de las columnas de dframe

        output:
            -Archivo HTML que permite escuchar el audio
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
