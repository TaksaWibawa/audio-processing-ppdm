import librosa
import numpy as np
import matplotlib.pyplot as plt


def get_data(file):
    audio_data, sampling_rate = librosa.load(file)
    return audio_data, sampling_rate
