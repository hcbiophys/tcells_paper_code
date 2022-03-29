from scipy import interpolate
import matplotlib.pyplot as plt



class Consecutive_Frames():
    """
    Class for a section of continuous frames (with no large gaps)
    """


    def __init__(self, name, t_res_initial):
        """
        Args;
        - name: essentially the index of the cell, but letters are appended for each continuous time series section
        (since a few have big gaps in the time series). For example if cell 'CELL1' has two continuous sections with a gap, these have indices 'CELL1a' and 'CELL1b'.
        - t_res_initial: time resolution of frames before linear interpolation to 5s gaps
        """

        self.name = name
        self.t_res_initial = t_res_initial

        self.pca0_list = []
        self.pca1_list = []
        self.pca2_list = []
        self.speed_uropod_list = []
        self.speed_uropod_running_mean_list = []

        self.names_list = None

        self.spectrogram = None
        self.embeddings = None

        self.orig_frame_list = []
        self.closest_frames = []

        self.PC_uncertainties = None

        self.color = None


    def add(self, frame, pca0, pca1, pca2, speed_uropod, speed_uropod_running_mean):
        """
        Add to the attribute lists
        """

        self.orig_frame_list.append(frame)
        self.pca0_list.append(pca0)
        self.pca1_list.append(pca1)
        self.pca2_list.append(pca2)
        self.speed_uropod_list.append(speed_uropod)
        self.speed_uropod_running_mean_list.append(speed_uropod_running_mean)


    def interpolate(self):
        """
        Linear interpolation to 5s gaps
        """

        frame_times = [j*self.t_res_initial for j in self.orig_frame_list]

        new_frame_times = [frame_times[0]]
        i = 1
        while new_frame_times[0] + i*5 < max(frame_times):
            new_frame_times.append(new_frame_times[0] + i*5)
            i+=1

        for idx in range(len(new_frame_times)):
            dists = [abs(new_frame_times[idx]-i) for i in frame_times]
            closest_frame = self.orig_frame_list[dists.index(min(dists))]
            self.closest_frames.append(closest_frame)

        for attribute in ['pca0_list', 'pca1_list', 'pca2_list', 'speed_uropod_list', 'speed_uropod_running_mean_list']:

            f = interpolate.interp1d(frame_times, getattr(self, attribute))
            y_new = f(new_frame_times)
            setattr(self, attribute, y_new)
