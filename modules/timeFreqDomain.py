import librosa
import matplotlib.pyplot as plt
import numpy as np

plt.rc('axes', titlesize=20)

def getSpectogram(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik Spectogram
    stft = librosa.stft(y=audio_data)
    dbValue = librosa.amplitude_to_db(np.abs(stft), ref=np.max) # amplitude_to_db digunakan untuk mengubah nilai amplitude menjadi nilai dB

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.specshow(dbValue, sr=sampling_rate, x_axis='time', cmap='cool', ax=ax) # specshow digunakan untuk menampilkan spektogram
    ax.set(title="Spectogram Representation")
    plt.tight_layout()

    return fig