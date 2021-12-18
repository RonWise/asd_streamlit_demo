import os
import glob
import streamlit as st
import numpy as np
import pandas as pd

st.header('Summarizing')
df = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20)))

st.dataframe(df)

st.header('Sound examples')

exmp_dir = 'sound'
machines = [f.name for f in os.scandir(exmp_dir) if f.is_dir()]

for machine in machines:
    with st.expander(machine):
        col1, col2 = st.columns(2)
        examples = glob.glob(os.path.join(exmp_dir, machine, '*.wav'))
        with col1:
            st.markdown("Normal")
            file = examples[0] if 'normal' in examples[0] else examples[1]
            audio_file = open(file, 'rb')
            audio_bytes = audio_file.read()

            st.audio(audio_bytes, format='audio/wav')

        with col2:
            st.markdown("Anomaly")
            file = examples[0] if 'anomaly' in examples[0] else examples[1]
            audio_file = open(file, 'rb')
            audio_bytes = audio_file.read()

            st.audio(audio_bytes, format='audio/wav')


st.header('GAN')
exmp_dir = 'gan_example'
epoch = st.select_slider(
        'Select epoch to listen to synthesized sound',
        options=['0', '10', '100', '200', '500', '1000'])
for machine in machines:
    with st.expander(machine):
        gan_examples = glob.glob(os.path.join(exmp_dir, machine, '*.wav'))
        file = None
        for name in gan_examples:
            if epoch in name:
                file = name
                break
        audio_file = open(file, 'rb')
        audio_bytes = audio_file.read()
        st.markdown("Synthesised normal sound")
        st.audio(audio_bytes, format='audio/wav')