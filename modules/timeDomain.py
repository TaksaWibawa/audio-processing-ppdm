import librosa
import matplotlib.pyplot as plt
import numpy as np

plt.rc('axes', titlesize=20)

def getAmplitudeTime(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Amplitude-Time
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate)
    ax.set(title="Amplitude-Time Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Amplitude")
    return fig

def getAverageEnergy(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Average Energy
    energy = librosa.feature.rms(y=audio_data)[0] # rms digunakan untuk menghitung nilai rata-rata dari energy audio
    
    frames = range(0, len(energy))
    time = librosa.frames_to_time(frames, sr=sampling_rate)
    
    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, energy, color='r', label='Average Energy')
    ax.set(title="Average Energy Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Energy")
    return fig

def getZeroCrossingRate(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)[0] # zero_crossing_rate digunakan untuk menghitung nilai rata-rata dari zero crossing rate

    frames = range(0, len(zcr))
    time = librosa.frames_to_time(frames, sr=sampling_rate)

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, zcr, color='r', label="Zero Crossing Rate")
    ax.set(title="Zero Crossing Rate Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Zero Crossing Rate")
    return fig

def getSilenceRatio(audio_data, sampling_rate):
    energy = librosa.feature.rms(y=audio_data)[0]
    zcr = librosa.feature.zero_crossing_rate(y=audio_data)[0]

    silenceRatio = np.sum(energy < max(energy)*0.1) / float(len(zcr)) # silenceRatio digunakan untuk menghitung nilai rasio dari silence

    frames = range(0, len(energy))
    time = librosa.frames_to_time(frames, hop_length=512)

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.waveshow(audio_data, sr=sampling_rate, ax=ax)
    ax.plot(time, energy, color='r', label='Silence Ratio')
    ax.set(title="Silence Ratio Representation")
    ax.set(xlabel="Time (s)")
    ax.set(ylabel="Energy")

    return fig, silenceRatio
