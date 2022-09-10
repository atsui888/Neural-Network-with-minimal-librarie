import numpy as np
import pickle


class Normalise:
    def __init__(self, input_data):
        self._matrix = input_data
        if self._matrix.ndim == 1:
            self._matrix = self._matrix.reshape(-1, 1)

        # stores min (row 0), max (row 1) of each column of input_data
        self._matrix_min_max = np.zeros((2, self._matrix.shape[1]), dtype=float)
        # print(self._matrix_min_max.shape)

        self._normalised_matrix = None

    def fit(self):
        # get the min and max for each col of self._matrix
        for col_num in range(self._matrix.shape[1]):
            self._matrix_min_max[0, col_num] = self._matrix[:, col_num].min()
            self._matrix_min_max[1, col_num] = self._matrix[:, col_num].max()
        self.save_min_max_matrix()
        return self

    def get_min_max_matrix(self):
        return self._matrix_min_max

    def save_min_max_matrix(self, file_name='normalise_min_max.pkl'):
        with open(file_name, 'wb') as handle:
            pickle.dump(self._matrix_min_max, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_min_max_matrix(self, file_name='normalise_min_max.pkl'):
        with open(file_name, 'rb') as handle:
            self._matrix_min_max = pickle.load(handle)

    def transform(self, data=None):
        """
        :param data:
        :return:
        """
        if data is not None:
            self._matrix = data

        for col_num in range(self._matrix.shape[1]):
            tmp = self.vec_normalise(self._matrix[:, col_num],
                                     self._matrix_min_max[0, col_num], self._matrix_min_max[1, col_num])
            if col_num == 0:
                self._normalised_matrix = tmp
            else:
                self._normalised_matrix = np.column_stack((self._normalised_matrix, tmp))
        return self._normalised_matrix

    def get_normalised_matrix(self):
        return self._normalised_matrix

    def run_normalise(x, x_min, x_max, rounding=4):
        return np.round((x - x_min) / (x_max - x_min), rounding)

    vec_normalise = np.vectorize(run_normalise)


if __name__ == '__main__':
    test_1d = np.array([1, 2, 4, 5, 3])
    test_2d = np.array([[1, 2], [2, 4], [4, 8], [5, 10], [3, 6]])
    test_3d = np.array([[1, 2, 4], [2, 4, 8], [4, 8, 12], [5, 10, 15], [3, 6, 9]])

    print(test_3d)
    print()
    N = Normalise(test_3d)
    N.fit().transform()
    print(N.get_min_max_matrix())
    print()
    print(N.get_normalised_matrix())