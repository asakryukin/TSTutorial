from utils.Model import Model
from utils.DataHandler import DataHandler

DH =DataHandler("Data/mnist_train.csv")
NN = Model()

NN.train(DH,10000)
