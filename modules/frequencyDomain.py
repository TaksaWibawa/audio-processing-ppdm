import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

plt.rc('axes', titlesize=20)

def getSoundSpectrum(audio_data, sampling_rate):
    stft = np.abs(librosa.stft(y=audio_data))
    
    dbValue = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    dbAverage = np.mean(dbValue, axis=1)
    
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    ax.bar(np.arange(0, len(dbAverage)), height=dbAverage, edgecolor='#788feb')

    x_ticks_position = [n for n in range(0, 2048 // 2, 2048 // 16)]
    x_ticks_labels = [str(int(round(sampling_rate / 2048 * n,0))) for n in x_ticks_position]
    ax.set_xticks(x_ticks_position, x_ticks_labels, rotation=45)

    ax.set(title="Sound Spectrum Representation")
    ax.set(xlabel="Frequency (Hz)")
    ax.set(ylabel="dB")
    ax.invert_yaxis()
    return fig

def getBandwidth(audio_data, sampling_rate):
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sampling_rate,  center=True, pad_mode='reflect')[0]

    frames = range(0, len(bandwidth))
    time = librosa.frames_to_time(frames)
    
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, minmax_scale(bandwidth, axis=0), color='r', label='Bandwidth')
    ax.set(title="Bandwidth Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Amplitude")
    return fig

def getSpectralCentroid(audio_data, sampling_rate):
    spectralCentroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate, center=True, pad_mode='reflect')[0]

    frames = range(0, len(spectralCentroid))
    time = librosa.frames_to_time(frames)

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, minmax_scale(spectralCentroid, axis=0), color='r', label='Spectral Centroid')
    ax.set(title="Spectral Centroid Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Amplitude")
    return fig