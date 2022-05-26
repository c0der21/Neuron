import pandas as pd
import numpy as np
from PIL import Image


def matx_to_vec(matx):
    vec = np.zeros(matx.shape[0] * matx.shape[1])
    k = 0
    for i in matx.index:
        for j in matx.columns:
            vec[k] = matx.at[i, j]
            k += 1
    return vec


def vec_to_mat(vec, n_cols=8):
    matx = np.zeros((int(len(vec) / n_cols), n_cols))
    k = 0
    for i in range(matx.shape[0]):
        for j in range(matx.shape[1]):
            matx[i, j] = vec[k]
            k += 1
    return matx


def matx_to_img(matrix):
    y = np.zeros(matrix.shape, dtype=np.uint8)
    y[matrix == 1] = 255
    y[matrix == -1] = 0
    img = Image.fromarray(y, mode="L")
    img.save("neuron.png")


class Neuron:
    def __init__(self, train, test, sample_count):
        self.train = train
        self.test = matx_to_vec(test)
        self.sample_c = sample_count
        self.weights = []
        self.axons_old = matx_to_vec(test)
        self.axons_new = np.zeros(test.shape[1] * test.shape[0])

    def axons_hopfield(self):
        k = 0
        for cond in self.weights.dot(self.axons_old):
            if cond < 0:
                self.axons_new[k] = -1
            else:
                self.axons_new[k] = 1
            k += 1

    def hidden_hamming(self):
        self.axons_old = np.zeros(self.weights.shape[0])
        self.axons_new = np.zeros(self.weights.shape[0])
        k = 0
        for cond in self.weights.dot(self.test):
            self.axons_old[k] = cond + self.weights.shape[1] / 2
            k += 1

    def axons_hamming(self, eps=0.01):
        for j in range(len(self.axons_old)):
            cond = self.axons_old[j] - eps * (sum(self.axons_old) - self.axons_old[j])
            if cond >= 0:
                self.axons_new[j] = cond
            else:
                self.axons_new[j] = 0

    def weights_hopfield(self):
        self.weights = np.zeros((len(self.test), len(self.test)))
        for i in range(self.weights.shape[0]):
            for j in range(i, self.weights.shape[1]):
                if i != j:
                    for k in range(self.sample_c):
                        self.weights[i][j] += self.train[k][i] * self.train[k][j]
                    self.weights[j][i] = self.weights[i][j]

    def hopfield(self):
        self.weights_hopfield()
        iteration = 0
        while iteration < 10000:
            self.axons_hopfield()
            iteration += 1
            if np.array_equal(self.axons_old, self.axons_new):
                break
            self.axons_old = self.axons_new
            self.axons_new = np.zeros(len(self.axons_old))
        self.axons_old = vec_to_mat(self.axons_old)
        matx_to_img(self.axons_old)

    def hamming(self):
        self.weights = self.train / 2
        self.hidden_hamming()
        iteration = 0
        while iteration < 10000:
            iteration += 1
            self.axons_hamming()
            if np.linalg.norm(np.array(self.axons_old) - np.array(self.axons_new)) < 0.1:
                break
            self.axons_old = self.axons_new
            self.axons_new = np.zeros(len(self.axons_old))
        print(self.axons_old)

    def start(self, n_rows, n_cols):
        models = np.empty((self.sample_c, n_rows*n_cols), int)
        for i in range(self.sample_c):
            models[i] = matx_to_vec(self.train.iloc[n_rows*i:n_rows*(i+1), :])
        self.train = models
        self.hopfield()
        self.hamming()


class_c = 4
n_row = 8
n_col = 8
data = pd.read_excel('examples.xlsx', usecols="B:I", nrows=class_c*n_row)
test_table = pd.read_excel('examples.xlsx', usecols="B:I", skiprows=class_c*n_row)

neuron = Neuron(data, test_table, class_c)
neuron.start(n_row, n_col)
