import sys
import logging

import librosa
import numpy as np


logger = logging.getLogger(__name__)


def file_load(wav_name, mono=False):
    """
    load .wav file.
    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    mono : boolean
        When load a multi channels file and this param True, the returned data will be merged for mono data
    return : numpy.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=mono)
    except:
        logger.error("file_broken or not exists!! : {}".format(wav_name))


def logmelspectrogram(
    file_name,
    n_fft=1024,
    hop_length=512,
    n_mels=128,
    power=2.0,
):
    y, sr = file_load(file_name)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
    )

    # 03 convert melspectrogram to log mel energy
    mel_spectrogram[mel_spectrogram < sys.float_info.epsilon] = sys.float_info.epsilon
    log_mel_spectrogram = 20.0 / power * np.log10(mel_spectrogram)

    return log_mel_spectrogram
