import librosa
import librosa.display

import matplotlib.pyplot as plt

from utils import logmelspectrogram


def plot_wav_melspectrogram(file):
    y, sr = librosa.load(file)

    fig, ax = plt.subplots(2, 1)
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    S = logmelspectrogram(file)
    librosa.display.specshow(S, x_axis='time', y_axis='log', ax=ax[1])

    ax[0].set(title='Sound wave')
    ax[1].set(title='Log-frequency power spectrogram')
    plt.tight_layout()
    return fig
