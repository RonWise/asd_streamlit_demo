import json
import os
import glob

import numpy
import numpy as np
from scipy import ndimage
import pandas as pd
import librosa
import librosa.display
import streamlit as st
from PIL import Image
import plotly.express as px

from plot_utils import plot_wav_melspectrogram

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

ERROR_MAPPER = {
    "normal-good-case": "normal_good",
    "anomaly-good-case": "anomaly_good",
    "normal-bad-case": "normal_bad",
    "anomaly-bad-case": "anomaly_bad",
}

image = Image.open("images/Header.png")
st.image(image)

st.header(
    "Sound examples",
)

exmp_dir = "sound"
machines = [f.name for f in os.scandir(exmp_dir) if f.is_dir()]

# EXAMPLE

object_type_example = st.selectbox(
    "Choose object type",
    ("valve", "slider", "ToyCar", "pump", "fan", "ToyConveyor"),
    key="example",
)
image = Image.open(f"images/object_img/{object_type_example}.png")
st.image(image, width=300, caption=f"{object_type_example} Image")

col1, col2 = st.columns(2)
examples = glob.glob(os.path.join(exmp_dir, object_type_example, "*.wav"))
with col1:
    st.markdown("Normal")
    file = examples[0] if "normal" in examples[0] else examples[1]
    # Plot Raw and MelSpectrogram
    fig = plot_wav_melspectrogram(file)
    st.pyplot(fig)
    # Plot Audio Bar
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

with col2:
    st.markdown("Anomaly")
    file = examples[0] if "anomaly" in examples[0] else examples[1]
    # Plot Raw and MelSpectrogram
    fig = plot_wav_melspectrogram(file)
    st.pyplot(fig)
    # Plot Audio Bar
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/wav")

# augmentations
aug_details = {
    "AddGaussianNoise": "Noise",  #: added noise helps against low audio signal quality",
    "FrequencyMask": "Frequency mask",
    #: masking blocks of consecutive frequency channels. It helps the model to be robust against partial loss of frequency information in the input.",
    "TimeMask": "Time mask",  #: helps against partial loss of small segments of signal.",
    "TimeStretch": "Time warping",  #: helps against deformations in the time direction.",
    "LowPassFilter": "Low-pass filter",
    "HighPassFilter": "High-pass filter",
}

st.header("Augmentations")
exmp_dir = "aug_example"
examples = glob.glob(os.path.join(exmp_dir, object_type_example, "normal_*.wav"))

col1, col2 = st.columns(2)

with col1:
    for file in examples[:3]:
        file_short_name = file.split("/")[-1].split(".")[0]
        # print augmentation details
        st.markdown(aug_details[file_short_name.split("_")[-1]])
        # Plot Raw and MelSpectrogram
        fig = plot_wav_melspectrogram(file)
        st.pyplot(fig)
        # Plot Audio Bar
        with open(file, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")

with col2:
    for file in examples[3:]:
        file_short_name = file.split("/")[-1].split(".")[0]
        # print augmentation details
        st.markdown(aug_details[file_short_name.split("_")[-1]])
        # Plot Raw and MelSpectrogram
        fig = plot_wav_melspectrogram(file)
        st.pyplot(fig)
        # Plot Audio Bar
        with open(file, "rb") as audio_file:
            audio_bytes = audio_file.read()
            st.audio(audio_bytes, format="audio/wav")

# # Error

st.header("Errors")
exmp_dir = "gan_example"

object_type_error = st.selectbox(
    "Choose object type",
    ("valve", "slider", "ToyCar", "pump", "fan", "ToyConveyor"),
    key="error",
)
with open(f"error_example/{object_type_error}.json") as json_file:
    data = json.load(json_file)

object_id_error = st.selectbox(
    "Choose object type",
    data.keys(),
    key="object_id_error",
)
case = st.selectbox(
    "Choose case",
    ERROR_MAPPER.keys(),
    key="case",
)
object_data_errors = data[object_id_error][ERROR_MAPPER[case]]
title = "Anomaly" if "anomaly" in case else "Normal"
title_case = "Good" if "good" in case else "Bad"
for i in range(0, 2):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(
        f'{title_case} reconstruction result for {title} file \n Anomaly Score: {round(object_data_errors[i]["score"], 2)}',
        y=0.80,
    )
    ax1.imshow(ndimage.rotate(object_data_errors[i]["x"], 90))
    ax1.set_title("X")
    ax1.axis("off")
    ax2.imshow(ndimage.rotate(object_data_errors[i]["y"], 90))
    ax2.set_title("Y")
    ax2.axis("off")
    ax3.imshow(ndimage.rotate(object_data_errors[i]["error"], 90))
    ax3.set_title(f"Error")
    ax3.axis("off")
    st.pyplot(fig)

# Latent Space

st.header("Latent Space")

object_type_latent = st.selectbox(
    "Choose object type",
    ("valve", "slider", "ToyCar", "pump", "fan", "ToyConveyor"),
    key="latent",
)

data = pd.read_csv(f"latent_space/{object_type_latent}.csv", index_col=0)

fig = px.scatter_3d(
    data, x="tsne-3d-one", y="tsne-3d-two", z="tsne-3d-three", color="status_id"
)

st.plotly_chart(fig, use_container_width=True)

# GAN

st.header("GAN")
exmp_dir = "gan_example"

object_type_gan = st.selectbox(
    "Choose object type",
    ("valve", "slider", "ToyCar", "pump", "fan", "ToyConveyor"),
    key="gan",
)
epoch = st.select_slider(
    "Select epoch to listen to synthesized sound",
    options=["150", "300", "450", "600", "750", "900", "1050", "1200", "1350", "1500"],
)
gan_dir = os.path.join(exmp_dir, object_type_gan)
for i in range(1, 3):
    origin_file = os.path.join(gan_dir, f"original_{i}.wav")
    synthesised_file = os.path.join(gan_dir, f"gan_{i}_{epoch}.wav")
    col1, col2 = st.columns(2)
    with col1:
        fig = plot_wav_melspectrogram(origin_file)
        st.pyplot(fig)
        with open(origin_file, "rb") as fin:
            audio_bytes = fin.read()
            st.markdown("Original Sound")
            st.audio(audio_bytes, format="audio/wav")
    with col2:
        fig = plot_wav_melspectrogram(synthesised_file)
        st.pyplot(fig)
        with open(synthesised_file, "rb") as fin:
            audio_bytes = fin.read()
            st.markdown(f"Synthesised normal sound on {epoch} epoch")
            st.audio(audio_bytes, format="audio/wav")

st.header("Summarizing")

# df = pd.read_csv("results/result.csv")
# st.dataframe(df)

df = pd.read_csv("results/form_result.csv", sep=";", dtype=float, header=None)
df.columns = pd.MultiIndex.from_product(
    [["ToyCar", "ToyConveyor", "Fan", "Pump", "Slider", "Valve"], ["pAUC", "AUC"]],
    names=["Device:", "Metric:"],
)
df.index = pd.Index(["0", "1", "2", "3", "Average"], name="#")


def style_negative(v):
    if v > 0.7:
        return "color: green;"
    if v > 0.6:
        return "color: orange;"
    return None


df = (
    df.style.applymap(style_negative)
    .applymap(lambda v: "opacity: 20%;" if (v < 0.5) else None)
    .format(na_rep="–", formatter="{:.2f}")  # precision=0,
)

# st.dataframe(df)

st.table(df)
