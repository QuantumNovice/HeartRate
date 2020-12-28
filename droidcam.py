from heartrate import HeartBeat
import cv2
import numpy as np
import matplotlib.pyplot as plt

import pickle
import statistics

from tqdm import tqdm

import requests

class DroidBeat(HeartBeat):
    '''
    Cam droid clients
    '''
    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)
        self.framerate = self.cap.get(cv2.CAP_PROP_FPS)


        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.brightness = []

        # Turn on LED
        x = requests.get('http://192.168.10.8:4747/cam/1/led_toggle')
        # print(x.status_code)

    def get_brightness_from_video(self):

        print("Image Processing")
        for i in tqdm(range(self.frames)):
            # print(i)
            ret, frame = self.cap.read()

            hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()

            self.brightness.append(sum(cdf_normalized) / len(cdf_normalized))

            # cv2.imshow('frame',frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # with open("test3.pkl", "wb") as afile:
        #     pickle.dump(self.brightness, afile)

        self.cap.release()
        cv2.destroyAllWindows()

        # plt.plot(self.brightness)
        # plt.show()

    def analyze_heart_beat(self):
        # print(self.framerate)
        # with open('test.pkl', 'rb') as afile:
        #     brightness = pickle.load(afile)

        # skip first few frames
        self.brightness = np.array(self.brightness[60:])

        self.brightness /= max(self.brightness)

        delta_time = 1 / self.framerate

        time_scale = np.arange(0, self.frames / self.framerate, delta_time)

        # print(time_scale)

        plt.plot(self.brightness/max(self.brightness) )
        # print(len(np.diff(brightness) ), len(brightness))

        locs = np.diff(self.brightness)

        clip_off = 0.0104
        locs = locs.clip(clip_off, 1)
        # plt.plot(locs)

        locii = np.where(locs != clip_off)
        time = locii[0] / self.framerate
        # print(locii[0])

        locii = locii[0]
        batch_size = 3

        R_counter = list(locii)
        for i in range(0, len(locii) - len(locii) % batch_size, batch_size):
            stdev = statistics.stdev(locii[i : i + batch_size])

            if stdev < 13:

                R_counter.remove(locii[i])
                # R_counter.remove(locii[i+batch_size])

        plt.show()
        return int(len(R_counter) * 60 / (self.frames / self.framerate) / 2)


if __name__ == '__main__':
    b = DroidBeat("http://192.168.10.8:4747/video")
    b.frames = 200

    print(f'framerate : {b.framerate}')

    b.get_brightness_from_video()

    b = b.analyze_heart_beat()
    print(b)