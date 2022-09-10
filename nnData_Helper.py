import numpy as np


class DataHelper:

    def normalise(x, x_min, x_max, rounding=4):
        return np.round((x - x_min) / (x_max - x_min), rounding)

    vec_normalise = np.vectorize(normalise)

    @staticmethod
    def normalise_matrix(matrix, matrix_std=None):
        """
        :param matrix:
        :param matrix_std: the matrix from which the min and max is to be used
        e.g. if train_set was normalised, then if matrix==test_test, then it needs to use the min and max
        which the train set was normalised,
        ie. normalise_matrix(test_input, train_input)
        :return:
        """
        # matrix: a np array
        if matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        if matrix_std is not None:
            if matrix_std.ndim == 1:
                matrix_std = matrix_std.reshape(-1, 1)

        normalised_arr = None
        # column-wise
        for i in range(len(matrix[0])):
            if matrix_std is not None:
                x_min = matrix_std[:, i].min()
                x_max = matrix_std[:, i].max()
            else:
                x_min = matrix[:, i].min()
                x_max = matrix[:, i].max()
            tmp = DataHelper.vec_normalise(matrix[:, i], x_min, x_max)
            if i == 0:
                normalised_arr = tmp
            else:
                normalised_arr = np.column_stack((normalised_arr, tmp))
        return normalised_arr


    @staticmethod
    def is_list_of_lists(container):
        if not isinstance(container, (list, tuple)):
            raise TypeError('Container must be of data type: List or Tuple.')
            # return False
        else:
            for idx, item in enumerate(container):
                if not isinstance(item, list):
                    msg = 'Items in container must be of data type "list"'
                    msg += f'Error item: index: {idx}.'
                    raise TypeError(msg)
                    return False
        return True

    @staticmethod
    def list_to_listoflists(data):
        return [[i] for i in data]
