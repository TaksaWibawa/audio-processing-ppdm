import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import minmax_scale

plt.rc('axes', titlesize=20)

def getSoundSpectrum(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Sound Spectrum
    stft = np.abs(librosa.stft(y=audio_data)) # stft digunakan untuk menghitung nilai Short Time Fourier Transform (STFT)
    
    dbValue = librosa.amplitude_to_db(np.abs(stft), ref=np.max) # amplitude_to_db digunakan untuk mengubah nilai amplitude menjadi nilai dB
    dbAverage = np.mean(dbValue, axis=1) # Menghitung nilai rata-rata dari dB
    
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    ax.bar(np.arange(0, len(dbAverage)), height=dbAverage, edgecolor='#788feb') # Membuat bar chart dari nilai rata-rata dB

    x_ticks_position = [n for n in range(0, 2048 // 2, 2048 // 16)] # Membuat posisi x ticks
    x_ticks_labels = [str(int(round(sampling_rate / 2048 * n,0))) for n in x_ticks_position] # Membuat label x ticks
    ax.set_xticks(x_ticks_position, x_ticks_labels, rotation=45) # Menampilkan x ticks

    ax.set(title="Sound Spectrum Representation")
    ax.set(xlabel="Frequency (Hz)")
    ax.set(ylabel="dB")
    ax.invert_yaxis()
    return fig

def getBandwidth(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Bandwidth
    # bandwidth digunakan untuk menghitung nilai bandwidth dari audio
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sampling_rate,  center=True, pad_mode='reflect')[0]

    frames = range(0, len(bandwidth))
    time = librosa.frames_to_time(frames)
    
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, minmax_scale(bandwidth, axis=0), color='r', label='Bandwidth') # minmax_scale digunakan untuk mengubah nilai bandwidth menjadi nilai antara 0 dan 1
    ax.set(title="Bandwidth Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Amplitude")
    return fig

def getSpectralCentroid(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Spectral Centroid
    # spectral_centroid digunakan untuk menghitung nilai spectral centroid dari audio
    spectralCentroid = librosa.feature.spectral_centroid(y=audio_data, sr=sampling_rate, center=True, pad_mode='reflect')[0]

    frames = range(0, len(spectralCentroid))
    time = librosa.frames_to_time(frames)

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, minmax_scale(spectralCentroid, axis=0), color='r', label='Spectral Centroid') # minmax_scale digunakan untuk mengubah nilai spectral centroid menjadi nilai antara 0 dan 1
    ax.set(title="Spectral Centroid Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Amplitude")
    return fig