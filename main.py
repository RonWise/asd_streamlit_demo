import os
import glob

import numpy as np
import pandas as pd
import librosa
import librosa.display
import streamlit as st
from PIL import Image

from plot_utils import plot_wav_melspectrogram

import matplotlib.pyplot as plt

image = Image.open('images/Header.png')
st.image(image)

st.header('Sound examples', )

exmp_dir = 'sound'
machines = [f.name for f in os.scandir(exmp_dir) if f.is_dir()]

# EXAMPLE

object_type_example = st.selectbox(
    'Choose object type',
    ('valve', 'slider', 'ToyCar', 'pump', 'fan', 'ToyConveyor'),
    key='example'
)
image = Image.open(f'images/object_img/{object_type_example}.png')
st.image(image, width=300, caption=f'{object_type_example} Image')

col1, col2 = st.columns(2)
examples = glob.glob(os.path.join(exmp_dir, object_type_example, '*.wav'))
with col1:
    st.markdown("Normal")
    file = examples[0] if 'normal' in examples[0] else examples[1]
    # Plot Raw and MelSpectrogram
    fig = plot_wav_melspectrogram(file)
    st.pyplot(fig)
    # Plot Audio Bar
    with open(file, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

with col2:
    st.markdown("Anomaly")
    file = examples[0] if 'anomaly' in examples[0] else examples[1]
    # Plot Raw and MelSpectrogram
    fig = plot_wav_melspectrogram(file)
    st.pyplot(fig)
    # Plot Audio Bar
    with open(file, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

# GAN

st.header('GAN')
exmp_dir = 'gan_example'

object_type_gan = st.selectbox(
    'Choose object type',
    ('valve', 'slider', 'ToyCar', 'pump', 'fan', 'ToyConveyor'),
    key='gan'
)
epoch = st.select_slider(
    'Select epoch to listen to synthesized sound',
    options=['160', '320', '390', '640', '730', '840', '980', '1060', '1130', '1450'])
gan_dir = os.path.join(exmp_dir, object_type_gan)
origin_file = os.path.join(gan_dir, 'original_1.wav')
synthesised_file = os.path.join(gan_dir, f'gan_1_{epoch}.wav')
col1, col2 = st.columns(2)
with col1:
    fig = plot_wav_melspectrogram(origin_file)
    st.pyplot(fig)
    with open(origin_file, 'rb') as fin:
        audio_bytes = fin.read()
        st.markdown("Original Sound")
        st.audio(audio_bytes, format='audio/wav')
with col2:
    fig = plot_wav_melspectrogram(synthesised_file)
    st.pyplot(fig)
    with open(synthesised_file, 'rb') as fin:
        audio_bytes = fin.read()
        st.markdown(f"Synthesised normal sound on {epoch} epoch")
        st.audio(audio_bytes, format='audio/wav')



st.header('Summarizing')
df = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(df)
