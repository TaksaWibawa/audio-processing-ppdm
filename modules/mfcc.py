import librosa
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, scale

plt.rc('axes', titlesize=20)
scaler = StandardScaler() # scaler digunakan untuk mengubah nilai menjadi nilai antara 0 dan 1

def getMFCC(audio_data, sampling_rate): # Fungsi digunakan untuk menampilkan grafik MFCC
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sampling_rate, n_mfcc=13)

    fig, ax = plt.subplots(1, 1, figsize=(25, 10))
    librosa.display.specshow(scale(scaler.fit_transform(mfcc),axis=1), sr=sampling_rate, x_axis='time', cmap='cool', ax=ax) # specshow digunakan untuk menampilkan MFCC
    ax.set(title="MFCC Representation")
    return fig