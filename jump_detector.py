import json
import tarfile
from argparse import ArgumentParser
from os.path import abspath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class JumpDetector(object):
    def __init__(self, dataset_path, threshold=55, trimming_duration_seconds=5, **kwargs):
        """
        An object detecting jumps in accelerometer data
        :param dataset_path: Path to a dataset tarfile containing a data.csv file of data recorded with phybox,
        a supervised.json file containing jump timing for visual verification of detected jumps
        :param threshold: The value above which a jump will be detected in the norm of the accelerometer. In m/s^2
        :param trimming_duration_seconds: The duration in seconds of accelerometer data that will be removed.
        It's the time it takes to put your smartphone in your pocket, and to take it back and stop recording.
        Set to 5 seconds by default
        :param kwargs: Other kwargs parameters forwarded to pandas read_csv function. Used to read data.csv file
        """
        self.trimming_duration_seconds = trimming_duration_seconds
        self.dataset_path = abspath(dataset_path)
        self.threshold = threshold
        self.pandas_read_kwargs = kwargs

        self.accelerometer_data = None  # The accelerometer data in numpy format
        self.supervised = None  # Information of the jumps from a recorded video

        self.load_data()  # Load data and fill self.accelerometer_data and self.supervised

    def load_data(self):
        """
        Load data and fill self.accelerometer_data and self.supervised
        :return: None
        """
        with tarfile.open(self.dataset_path) as tar:
            for member in tar:
                name = member.name
                fp = tar.extractfile(member)
                if name == "data.csv":
                    # Extract pandas data
                    pandas_data = pd.read_csv(fp, **self.pandas_read_kwargs)
                elif name == "supervised.json":
                    self.supervised = json.load(fp)

        xyz = pandas_data[["Acceleration x (m/s^2)", "Acceleration y (m/s^2)", "Acceleration z (m/s^2)"]].values
        t = pandas_data["Time (s)"].values

        # Samplig rate
        Ts = np.diff(t).mean()

        # Number of samples to trimm
        Ntrim = int(np.round(self.trimming_duration_seconds / Ts))
        t = t[Ntrim:-Ntrim]
        xyz = xyz[Ntrim: -Ntrim, :]

        self.accelerometer_data = t, xyz

    @property
    def norm(self):
        """
        Compute the norm of accelerometer
        :return: time and norm both as numpy array
        """
        t, xyz = self.accelerometer_data
        return t, np.linalg.norm(xyz, axis=1)

    def jump_time(self, threshold=None):
        """
        Compute the jump timing
        :param threshold: Used to override threshold defined during class instanciation.
        :return: A numpy array giing the time at each detected jump occurred. (In the time base of the accelerometer)
        """
        t, norm = self.norm

        # Find the jumps using the norm of acceleration. When the user lands on its feet, the shock
        # of the feet touching the ground create a spike in the norm of the acceleration and thus a simple threshold
        # is enough to detect it. When more data get collected algorithm could be improved by pattern matching:
        # using a convolution of a kernel with the signal should improve the result.
        thresholded = norm > (threshold or self.threshold)
        # The jumps are the moment the norm exceed the threshold
        selector = np.diff(thresholded.astype(np.int8), prepend=0) > 0
        # From boolean selector, extract the jumping time
        jump_time = t[selector]

        # Remove jumps that are too close (less than half a second close one another)
        selector = np.diff(jump_time, prepend=-np.inf) > 0.5
        jump_time = jump_time[selector]

        return jump_time

    def plot(self):
        """
        Plot detected jumps & supervised jumps for visual comparison.
        (Timebase is the timebase of the video for verification)
        :return: ax of the matplotlib plot
        """
        # Get supervised info
        video_offset = self.supervised["video_offset"]
        supervised_jumps = np.array(self.supervised["jumps_timing_video"])

        # Get data to plot
        t, norm = self.norm
        video_time = t - video_offset
        jumps = self.jump_time()

        # Create plot
        fig, ax = plt.subplots(1, 1)

        # Plot jumps
        for jumps_to_print, color, ls, label in zip([supervised_jumps, jumps - video_offset],
                                                    ["C1", "g"],
                                                    ['solid', ':'],
                                                    "Supervised jump,Jump detected".split(",")):
            for i, jump in enumerate(jumps_to_print):
                settings = dict(color=color, ls=ls, lw=2)
                if i == 0:
                    settings["label"] = label
                ax.axvline(jump, **settings)

        # Plot signal
        ax.plot(video_time, norm, label="Acc norm")

        # Plot legend, etc
        ax.set_xlabel("Video time (seconds)")
        ax.set_ylabel("Acceleration (g)")
        ax.grid()
        ax.legend()
        ax.set_title("Visual verification of jump detection")
        return ax


def main():
    """Main function. Read CLI arguments and plot jump detection"""
    # Parse cli arguments
    parser = ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("-d", "--delimiter", type=str, default=";")
    args = parser.parse_args()

    # Create instance
    jd = JumpDetector(**args.__dict__)

    # Plot jumps
    jd.plot()
    plt.show()


if __name__ == '__main__':
    main()
