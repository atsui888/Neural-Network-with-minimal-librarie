import numpy as np
from nnData_Helper import DataHelper


class Layer:
    layer_type = 'Base Layer'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer):
        self.layer_name = layer_name
        self._nodes_prev_layer = nodes_prev_layer
        self._nodes = nodes_in_layer
        self._layer_matrix = None

    def get_layer_details(self):
        layer_details = {
            'name': self.layer_name,
            'layer_type': self.layer_type,
            'num_nodes_prev': self._nodes_prev_layer,
            'num_nodes': self._nodes
        }
        if self._layer_matrix is not None:
            layer_details['layer_shape'] = self._layer_matrix.shape
            layer_details['layer_matrix'] = self._layer_matrix
        return layer_details

    def print_layer_details(self):
        layer_details = self.get_layer_details()
        print('*' * 50)
        print(f"name: \t\t\t{layer_details['name']}")
        print(f"layer_type: \t{layer_details['layer_type']}")
        print(f"num_nodes_prev: {layer_details['num_nodes_prev']}")
        print(f"num_nodes: \t\t{layer_details['num_nodes']}")
        layer_shape = layer_details.get('layer_shape', 'Data not loaded yet')
        print(f"layer_shape: \t{layer_shape}")

    # def forward(self, X=None):
    #     # stub fn, implementation is to be overriden in derived classes if required.
    #     pass

    def get_layer_matrix(self):
        return self._layer_matrix


class InputLayer(Layer):
    """
    Input layer has no weights.
    """
    layer_type = 'Input Layer'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer):
        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer)
        self._data = None

    def set_data(self, data):
        data = DataHelper.list_to_listoflists(data)

        if DataHelper.is_list_of_lists(data):
            self._data = data

        self._layer_matrix = np.array(data).reshape(-1, self._nodes)  # representation of this layer


class FullyConnectedLayer(Layer):
    layer_type = 'Fully Connected Layer'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, act_fn):
        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer)

        self._weights_matrix = np.full(self._nodes_prev_layer * self._nodes, fill_value=0.1)
        # Todo: Uncomment next line when testing is over, coz we don't want the weights with a fixed number
        # self._weights_matrix = np.random.rand(self._nodes_prev_layer * self._nodes)
        self._weights_matrix = self._weights_matrix.reshape(self._nodes_prev_layer, self._nodes)

        self._layer_matrix = np.zeros(self._nodes).reshape(1, self._nodes)  # representation of this layer

        self.act_fn = act_fn

    def get_layer_details(self):
        layer_details = super().get_layer_details()
        layer_details['weights_shape'] = self._weights_matrix.shape
        layer_details['weights_matrix'] = self._weights_matrix
        layer_details['activation'] = self.act_fn.activation_type
        return layer_details

    def print_layer_details(self):
        super().print_layer_details()
        layer_details = self.get_layer_details()
        print(f"weights_shape: \t{layer_details['weights_shape']}")
        print(f"activation: \t{layer_details['activation']}")

    def get_weights_matrix(self):
        return self._weights_matrix

    def update_weights_matrix(self, learning_rate, weight_deltas):
        # once delta is 0, +=0 means no change in weights
        self._weights_matrix -= learning_rate * weight_deltas

    def forward(self, x):
        """
        _layer_matrix of this layer = dot pdt of the matrix of the prev layer (input) and the weights of this layer,
                                     after passing the result of the dot product through an activation function.
        We only calculate what the weights of this layer should be, no prediction is done here.
        :param x: data from previous layer
        :return:
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # .sum( axis=0) collapses the rows into 1 row
        # self._layer_matrix = np.sum(np.dot(X, self._weights_matrix), axis=0)
        self._layer_matrix = np.dot(x, self._weights_matrix)

        # run result through the activation function
        self._layer_matrix = self.act_fn.execute(self._layer_matrix)

        if self._layer_matrix.ndim == 1:
            self._layer_matrix = self._layer_matrix.reshape(1, -1)

    def test(self):
        self.act_fn.execute(888)


class OutputBinaryClassification(FullyConnectedLayer):
    layer_type = 'Output_Binary_Classification'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, act_fn):
        nodes_in_layer = 1  # hardcode coz binary classification

        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer, act_fn)

        self._predicted_class = None

    def get_probability_matrix(self):
        return self._layer_matrix

    def predict(self, threshold=0.5):
        # returns an array of array, hence [0][0] to get the int

        # both the below are the same, but the .astype() ver is supposed to be twice as fast
        # but ver B can only produce 0 and 1 while ver A is more flexible
        # self._predicted_class = np.where(self._layer_matrix >= threshold, 1, 0)]  # ver A
        self._predicted_class = (self._layer_matrix > threshold).astype(int)  # ver B
        # nb: .round() is the slowest, in some cases 10 times slower
        # ver A and B are faster than list comprehension coz they are vectorized.
        return self._predicted_class


# not implemented, act fn sh be softmax
# class Output_MultiClass_Classification(Fully_Connected_Layer): # predict 1 class out of multiple classes
# class Output_MultiLabel_Classification(Fully_Connected_Layer): # predict p-values of 1 or more classes
#   an input can belong to >1 class

class OutputRegression(FullyConnectedLayer):
    layer_type = 'Output_Regression'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, act_fn):
        nodes_in_layer = 1  # hardcode coz regression, the pred is a scaler, hence only one node
        self._prediction = 0.0

        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer, act_fn)

    def predict(self):
        """
        regression prediction is linear(wx + b), and this is ALREADY done in the forward
        pass.
        Hence, to 'predict', all we need to do here is to return the layer_matrix that was
        the result of the forward pass.
        :return:
        """
        return self._layer_matrix

