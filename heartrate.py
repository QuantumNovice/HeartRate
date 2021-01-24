import statistics

import cv2
import numpy as np

from tqdm import tqdm



class HeartBeat:
    """
    Computes heart rate from stream.
    """

    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.framerate = self.cap.get(cv2.CAP_PROP_FPS)

        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.brightness = []

    def get_brightness_from_video(self):
        """
        Calculates mean image intensity
        """
        print("Image Processing")
        for i in tqdm(range(self.frames)):
            # print(i)
            ret, frame = self.cap.read()

            hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            self.brightness.append(sum(cdf_normalized) / len(cdf_normalized))

        self.cap.release()
        cv2.destroyAllWindows()

    def analyze_heart_beat(self):
        """
        Perform statistical analysis on data to
        obtain heartrate.

        Steps are:
        * Normalize spatial/temporal
        * Find derivative
        * Make a null hypothesis
        * Use derivative to search for possible beats
          that are within null hypothesis.

        """

        self.brightness = np.array(self.brightness)

        self.brightness /= max(self.brightness)

        # delta_time = 1 / self.framerate

        # time_scale = np.arange(0, self.frames / self.framerate, delta_time)

        locs = np.diff(self.brightness)

        clip_off = 0.0104
        locs = locs.clip(clip_off, 1)

        locii = np.where(locs != clip_off)
        # time = locii[0] / self.framerate

        locii = locii[0]
        batch_size = 3

        R_counter = list(locii)
        for i in range(0, len(locii) - len(locii) % batch_size, batch_size):
            stdev = statistics.stdev(locii[i : i + batch_size])

            if stdev < 13:

                R_counter.remove(locii[i])

        return int(len(R_counter) * 60 / (self.frames / self.framerate) / 2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import requests

    import pickle
    b = HeartBeat("VID_20201025_204835.3gp")

    # b.frames = 50

    print(f"framerate : {b.framerate}")

    b.get_brightness_from_video()

    b = b.analyze_heart_beat()
    print(b)
