import numpy as np

class DataHandler:

    def __init__(self, filename):

        self.load_data(filename)
        self.split_data()


    def load_data(self,filename):
        self.data = []
        self.labels = []
        with open(filename, 'r') as file:
            next(file)
            for line in file:
                data = map(float,line.split(','))
                self.data.append(data[1:])
                self.labels.append(int(data[0]))

    def split_data(self):

        inds = np.random.choice(len(self.data),len(self.data))

        self.train_data = (np.reshape(np.array([self.data[i] for i in inds[0:int(0.9*len(inds))]]),[-1,28,28,1]))/255.0
        self.train_labels = np.reshape(np.array([self.labels[i] for i in inds[0:int(0.9*len(inds))]]),[-1])

        self.valid_data = np.reshape(np.array([self.data[i] for i in inds[int(0.9 * len(inds)):]]),[-1,28,28,1])/255.0
        self.valid_labels = np.reshape(np.array([self.labels[i] for i in inds[int(0.9 * len(inds)):]]),[-1])

    def get_training_batch(self,n):

        inds = np.random.choice(len(self.train_data), n)

        return [self.train_data[i] for i in inds],[self.train_labels[i] for i in inds]

