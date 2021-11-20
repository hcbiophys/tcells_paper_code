from scipy import interpolate
import matplotlib.pyplot as plt



class Consecutive_Frames():


    def __init__(self, name, t_res_initial):

        self.name = name
        self.t_res_initial = t_res_initial

        self.pca0_list = []
        self.pca1_list = []
        self.pca2_list = []
        self.RI_vector0_list = []
        self.delta_centroid_list = []
        self.delta_sensing_direction_list = []
        self.run_list = []
        self.run_mean_list = []
        self.spin_vec_magnitude_list = []
        self.spin_vec_magnitude_mean_list = []
        self.spin_vec_std_list = []
        self.direction_std_list = []

        self.spectrogram = None
        self.embeddings = None

        self.orig_frame_list = []
        self.closest_frames = []


    def add(self, frame, pca0, pca1, pca2, RI_vector0, delta_centroid, delta_sensing_direction, run, run_mean,  spin_vec_magnitude, spin_vec_magnitude_mean, spin_vec_std, direction_std):
        self.orig_frame_list.append(frame)
        self.pca0_list.append(pca0)
        self.pca1_list.append(pca1)
        self.pca2_list.append(pca2)
        self.RI_vector0_list.append(RI_vector0)
        self.delta_centroid_list.append(delta_centroid)
        self.delta_sensing_direction_list.append(delta_sensing_direction)
        self.run_list.append(run)
        self.run_mean_list.append(run_mean)

        self.spin_vec_magnitude_list.append(spin_vec_magnitude)
        self.spin_vec_magnitude_mean_list.append(spin_vec_magnitude_mean)
        self.spin_vec_std_list.append(spin_vec_std)
        self.direction_std_list.append(direction_std)


    def interpolate(self):

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

        for attribute in ['pca0_list', 'pca1_list', 'pca2_list', 'RI_vector0_list', 'delta_centroid_list', 'delta_sensing_direction_list', 'run_list', 'run_mean_list', 'spin_vec_magnitude_list', 'spin_vec_magnitude_mean_list', 'spin_vec_std_list', 'direction_std_list']:

            f = interpolate.interp1d(frame_times, getattr(self, attribute))
            y_new = f(new_frame_times)

            #plt.plot(frame_times, getattr(self, attribute), color = 'red')
            setattr(self, attribute, y_new)
            #plt.plot(new_frame_times, getattr(self, attribute), color = 'blue')
            #plt.show()
