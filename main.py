# 741 - 750
import streamlit as st
import pandas as pd
import os

from modules.getData import get_data
from modules.timeDomain import *
from modules.frequencyDomain import *
from modules.mfcc import *
from modules.timeFreqDomain import *

# Get the audio files from folder
parent_folder = "dataset/"
subfolder_names = ["Happy", "Neutral", "Sad"]

if __name__ == '__main__':
    st.title("Audio Processing")
    st.markdown(f'''
                    Nama  : I Made Sudarsana Taksa Wibawa
                    \nNIM   : 2108561109
                    \nKelas : D
                    \nLink Github : 
                    <a href="https://github.com/TaksaWibawa/audio-processing-ppdm">https://github.com/TaksaWibawa/audio-processing-ppdm</a>
                    \n-----
                ''', unsafe_allow_html=True)

    # Menampilkan daftar gambar apa saja yang ada
    st.header("Daftar Audio")
    df = pd.DataFrame(columns=['Audio Name', 'Category'])
    df_list = []
    for subfolder in subfolder_names:
        subfolder_path = os.path.join(parent_folder, subfolder)
        audio_list = os.listdir(subfolder_path)
        audio_names = [os.path.splitext(audio)[0] for audio in audio_list]
        category = [subfolder] * len(audio_names)
        audio_df = pd.DataFrame(
            {"Audio Name": audio_names, "Category": category})
        df_list.append(audio_df)
    df = pd.concat(df_list, ignore_index=True)
    st.dataframe(df, width=800, height=300)

    # Memuat keseluruhan audio files yang nantinya digunakan pada widget multiselect
    aud_list = []
    for subfolder in subfolder_names:
        subfolder_path = os.path.join(parent_folder, subfolder)
        aud_list += os.listdir(subfolder_path)
    st.write("Pilih File Audio yang Ingin Dianalisis")
    selected_aud = st.multiselect(
        label="Ketentuan : 3 Audio Happy, Neutral, dan Sad", options=aud_list, default=None)

    # Memproses audio yang dipilih
    if (st.button("Proses") and selected_aud):
        counter = 1
        for aud in selected_aud:
            for subfolder in subfolder_names:
                subfolder_path = os.path.join(parent_folder, subfolder)
                aud_path = os.path.join(subfolder_path, aud)
                if os.path.isfile(aud_path):
                    st.write("-----")
                    st.write(f"# Audio ke-{counter} : {aud}")
                    counter += 1
                    st.write("Kategori : " + subfolder)
                    st.audio(aud_path)

                    # Mengambil data audio
                    audio_data, sampling_rate = get_data(aud_path)

                    # Melakukan time domain
                    # 1. amplitude-time representation
                    st.write(f"#### 1. Amplitude-Time Representation")
                    fig_amplitude = getAmplitudeTime(audio_data, sampling_rate)
                    st.pyplot(fig_amplitude)

                    # 2. average energy
                    st.write(f"#### 2. Average Energy Representation")
                    fig_energy = getAverageEnergy(audio_data, sampling_rate)
                    st.pyplot(fig_energy)

                    # 3. zero crossing rate
                    st.write(f"#### 3. Zero Crossing Rate Representation")
                    fig_zcr = getZeroCrossingRate(audio_data, sampling_rate)
                    st.pyplot(fig_zcr)

                    # 4. silence ratio
                    st.write(f"#### 4. Silence Ratio Representation")
                    fig_silenceRatio, silenceRatio = getSilenceRatio(audio_data, sampling_rate)
                    st.write(f"###### Silence Ratio : " + str(silenceRatio))
                    st.pyplot(fig_silenceRatio)

                    # Melakukan frequency domain
                    # 1. sound spectrum representation
                    st.write(f"#### 5. Sound Spectrum Representation")
                    fig_soundSpectrum = getSoundSpectrum(audio_data, sampling_rate)
                    st.pyplot(fig_soundSpectrum)

                    # 2. bandwidth
                    st.write(f"#### 6. Bandwidth Representation")
                    fig_bandwidth = getBandwidth(audio_data, sampling_rate)
                    st.pyplot(fig_bandwidth)

                    # 3. spectral centroid
                    st.write(f"#### 7. Spectral Centroid Representation")
                    fig_spectralCentroid = getSpectralCentroid(audio_data, sampling_rate)
                    st.pyplot(fig_spectralCentroid)

                    # Melakukan ekstraksi fitur MFCC
                    st.write(f"#### 8. MFCC Representation")
                    fig_mfcc = getMFCC(audio_data, sampling_rate)
                    st.pyplot(fig_mfcc)

                    # Melakukan time frequency domain
                    # 1. spectrogram representation
                    st.write(f"#### 9. Spectrogram Representation")
                    fig_spectrogram = getSpectogram(audio_data, sampling_rate)
                    st.pyplot(fig_spectrogram)

    else:
        st.error("Tidak ada file audio yang dipilih!")
