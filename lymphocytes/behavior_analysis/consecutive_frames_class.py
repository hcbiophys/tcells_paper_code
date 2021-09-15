


class Consecutive_Frames():


    def __init__(self, name, t_res):

        self.name = name
        self.t_res = t_res
        
        self.frame_list = []
        self.pca0_list = []
        self.pca1_list = []
        self.pca2_list = []

        self.spectrogram = None
        self.embeddings = None


    def add(self, frame, pca0, pca1, pca2):
        self.frame_list.append(frame)
        self.pca0_list.append(pca0)
        self.pca1_list.append(pca1)
        self.pca2_list.append(pca2)
