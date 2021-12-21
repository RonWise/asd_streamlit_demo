import os
import glob
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from audiomentations import (
    AddGaussianNoise,
    TimeStretch,
    FrequencyMask,
    TimeMask,
    LowPassFilter,
    HighPassFilter,
)
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)

CURRENT_DIR = os.path.dirname(__file__)
APPLICATION_NAME = "apply_augm"
DEFAULT_INPUT_PATH = os.path.dirname(__file__)
DEFAULT_OUTPUT_PATH = os.path.join(CURRENT_DIR, "aug_output")


def process_files(output_dir, sound_file_paths):

    transforms = [
        {
            "instance": AddGaussianNoise(
                min_amplitude=0.001, max_amplitude=0.005, p=1.0
            ),
            "num_runs": 1,
        },
        {
            "instance": FrequencyMask(
                min_frequency_band=0.34, max_frequency_band=0.5, p=1.0
            ),
            "num_runs": 1,
        },
        {
            "instance": TimeMask(min_band_part=0.0, max_band_part=0.01, p=1.0),
            "num_runs": 1,
        },
        {
            "instance": TimeStretch(min_rate=0.5, max_rate=1.5, p=1.0),
            "num_runs": 1,
        },
        {
            "instance": LowPassFilter(min_cutoff_freq=150, max_cutoff_freq=7500, p=1.0),
            "num_runs": 1,
        },
        {
            "instance": HighPassFilter(min_cutoff_freq=20, max_cutoff_freq=2400, p=1.0),
            "num_runs": 1,
        },
    ]

    for sound_file_path in sound_file_paths:
        sound_file_name = sound_file_path.split(".wav")[0]
        samples, sample_rate = load_sound_file(
            sound_file_path, sample_rate=None, mono=False
        )
        if len(samples.shape) == 2 and samples.shape[0] > samples.shape[1]:
            samples = samples.transpose()

        print(f"Transforming {sound_file_path} with shape {str(samples.shape)}")

        for transform in transforms:
            augmenter = transform["instance"]
            run_name = (
                transform.get("name")
                if transform.get("name")
                else transform["instance"].__class__.__name__
            )

            for i in range(transform["num_runs"]):
                output_file_path = os.path.join(
                    output_dir,
                    f"{sound_file_name.split('/')[-1]}_{run_name}.wav",
                )
                try:
                    augmented_samples = augmenter(
                        samples=samples, sample_rate=sample_rate
                    )

                    if len(augmented_samples.shape) == 2:
                        augmented_samples = augmented_samples.transpose()

                    wavfile.write(
                        output_file_path, rate=sample_rate, data=augmented_samples
                    )
                except MultichannelAudioNotSupportedException as e:
                    print(e)


def process_pipeline(input_filepath, output_filepath):
    """
    Process augmentations
    """

    np.random.seed(42)
    random.seed(42)

    # import pdb; pdb.set_trace()

    output_dir = os.path.join(CURRENT_DIR, output_filepath)
    os.makedirs(output_dir, exist_ok=True)

    input_dir = os.path.join(CURRENT_DIR, input_filepath)
    sound_file_paths = glob.glob(os.path.join(input_dir, "*.wav"))
    process_files(output_dir, sound_file_paths)


def callback_parser(arguments):
    """Calling process function wth received arguments"""

    return process_pipeline(
        arguments.input_filepath,
        arguments.output_filepath,
    )


def setup_parser(parser):
    """Setting up parser"""

    parser.add_argument(
        "-i",
        "--input",
        default=DEFAULT_INPUT_PATH,
        metavar="INPUT",
        dest="input_filepath",
        help="path to audio files to process, default path is %(default)s",
    )

    parser.add_argument(
        "-o",
        "--ouput",
        default=DEFAULT_OUTPUT_PATH,
        metavar="OUTPUT",
        dest="output_filepath",
        help="path to output folder with augmented audio files, default path is %(default)s",
    )

    parser.set_defaults(callback=callback_parser)


def main():
    parser = ArgumentParser(
        prog="apply_augm",
        description="Script to generate audio augmentations",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
