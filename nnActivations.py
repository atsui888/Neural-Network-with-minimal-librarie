import numpy as np


class Softmax:
    activation_type = 'Softmax'
    # https://www.sharpsightlabs.com/blog/numpy-softmax/

    def execute(self, x):
        sm_distribution = (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())
        return sm_distribution


class Sigmoid:
    activation_type = 'Sigmoid'

    def execute(self, input_matrix):
        z = np.exp(-input_matrix)
        sig = 1 / (1 + z)
        return sig


class SigmoidStable:
    activation_type = 'Sigmoid_Stable'

    def execute(self, input_matrix):
        sig = np.where(input_matrix < 0, np.exp(input_matrix) / (1 + np.exp(input_matrix)),
                       1 / (1 + np.exp(-input_matrix)))
        return sig


class Linear:
    activation_type = 'Linear'

    def execute(self, input_matrix):
        return input_matrix * 1

