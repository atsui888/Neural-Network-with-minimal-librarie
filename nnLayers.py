import numpy as np
from nnData_Helper import DataHelper


class Layer:
    layer_type = 'Base Layer'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, debug_mode=False):
        self.layer_name = layer_name
        self._nodes_prev_layer = nodes_prev_layer
        self._nodes = nodes_in_layer

        """
        `self._layer_output` holds different types of values at different phases.
        Initially, it holds Wn.Xn
        then, Wn.Xn + b
        finally, when the entire forward pass is completed, it holds Act_fn(Wn.Xn + b)
        `self._layer_output` is different from the weights matrix and bias matrix which holds 
        weights and bias for that layer.
        """

        self._layer_weights = None  # declared here but initialised in derived classes
        self._layer_output = None  # declared here but value is stored here by methods in derived classes

        self._layer_bias = np.random.rand(1 * 1).reshape(-1, 1)  # these are bias "weights"

        self._layer_output_bias = None  # declared here but value is stored here by methods in derived classes

        self._debug_mode = debug_mode

    def get_layer_details(self):
        layer_details = {
            'name': self.layer_name,
            'layer_type': self.layer_type,
            'num_nodes_prev': self._nodes_prev_layer,
            'num_nodes': self._nodes
        }
        if self._layer_output is not None:
            layer_details['layer_output_shape'] = self._layer_output.shape
            layer_details['layer_output'] = self._layer_output
        return layer_details

    def print_layer_details(self):
        layer_details = self.get_layer_details()
        print('*' * 50)
        print(f"name: \t\t\t{layer_details['name']}")
        print(f"layer_type: \t{layer_details['layer_type']}")
        print(f"num_nodes_prev: {layer_details['num_nodes_prev']}")
        print(f"num_nodes: \t\t{layer_details['num_nodes']}")
        layer_output_shape = layer_details.get('layer_output_shape', 'No data, check your layer construction.')
        print(f"layer_output_shape: \t{layer_output_shape}")

    # def forward(self, X=None):
    #     # stub fn, implementation is to be overriden in derived classes if required.
    #     pass

    def get_layer_output(self):
        return self._layer_output

    def get_layer_output_bias(self):
        return self._layer_output_bias

    def get_layer_output_shape(self):
        return self._layer_output.shape


class InputLayer(Layer):
    """
    Inherits from base class Layer
    Input layer has no weights nor bias BUT has a matrix for data and another matrix for bias
    """
    layer_type = 'Input Layer'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, train_data, debug_mode=False):
        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer, debug_mode)
        self._data = None

        self.set_data_and_bias(train_data)

    def set_data_and_bias(self, data):
        self._data = data

        self._layer_output = np.array(data).reshape(-1, self._nodes)
        if self._layer_output.ndim == 1:
            self._layer_output = self._layer_output.reshape(-1, 1)

        # init bias output matrix to 0, origin
        self._layer_output_bias = np.zeros(self._data.shape[0] * 1).reshape(-1, 1)

        if self._debug_mode:
            print('\nInput Layer <-- set_data(self, data):')
            print(f"Number of Feature nodes in Input Layer:\t{self._nodes}")
            # if error, input data must be converted to numpy array first
            print(f"Incoming Training Data shape is:\t\t{data.shape}")
            print(f"'Input' Layer Output.shape is:\t\t\t{self._layer_output.shape}")
            """
            This NN implementation does not init weights at the input layer, but does init bias here
            so that we can do a dot product with the bias at the next hidden or output layer.
            In this manner, because N rows input data have N rows of bias, we can implement Multiprocessing
            in a simpler way.
            """
            print(f"'Input' Layer bias.shape is:\t\t\t{self._layer_bias.shape}")

        # data = DataHelper.list_to_listoflists(data)

        # if DataHelper.is_list_of_lists(data):
        #     self._data = data
        #     # num samples by num features (aka self._nodes)
        #     self._layer_matrix = np.array(data).reshape(-1, self._nodes)


class FullyConnectedLayer(Layer):
    layer_type = 'Fully Connected Layer'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, act_fn, debug_mode=False):
        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer, debug_mode)

        # Init Weights
        # ************
        # todo: comment out the next line when testing is over
        # self._layer_weights = np.full(self._nodes_prev_layer * self._nodes, fill_value=0.1)
        # todo: use randomised weights (below) after testing is done (comment out above line)
        self._layer_weights = np.random.rand(self._nodes_prev_layer * self._nodes)

        self._layer_weights = self._layer_weights.reshape(self._nodes_prev_layer, self._nodes)
        if self._layer_weights.ndim == 1:
            self._layer_weights = self._layer_weights.reshape(-1, 1)

        # Assign activation function
        self.act_fn = act_fn

        if debug_mode:
            print(f"Layer Name: {self.layer_name}")
            print(f"weights.shape:\t\t{self._layer_weights.shape}")
            print(f"bias.shape:\t\t{self._layer_bias.shape}")
            print(f"layer_output.shape:\t{self._layer_output.shape}")

    def get_layer_details(self):
        layer_details = super().get_layer_details()
        layer_details['weights_shape'] = self._layer_weights.shape
        layer_details['weights_matrix'] = self._layer_weights
        layer_details['activation'] = self.act_fn.activation_type
        return layer_details

    def print_layer_details(self):
        super().print_layer_details()
        layer_details = self.get_layer_details()
        print(f"weights_shape: \t{layer_details['weights_shape']}")
        print(f"activation: \t{layer_details['activation']}")

    def get_weights_matrix(self):
        return self._layer_weights

    def get_bias_matrix(self):
        return self._layer_bias

    def update_weights_bias(self, learning_rate, weight_deltas, bias_deltas):
        # once delta is 0, +=0 means no change in weights

        if weight_deltas.ndim == 1:
            weight_deltas = weight_deltas.reshape(-1, 1)

        if bias_deltas.ndim == 1:
            bias_deltas = bias_deltas.reshape(-1, 1)

        # print(f"self._weights_matrix.shape: {self._weights_matrix.shape}")
        # print(f"weight_deltas.shape: {weight_deltas.shape}")

        self._layer_weights -= learning_rate * weight_deltas
        self._layer_bias -= learning_rate * bias_deltas

    def forward(self, prev_layer_output, prev_layer_output_bias, debug_mode):
        """
        _layer_matrix of this layer = dot pdt of the matrix of the prev layer (input) and the weights of this layer,
                                     after passing the result of the dot product through an activation function.
        We only calculate what the weights of this layer should be, no prediction is done here.
        :param prev_layer_x: data from previous layer
        :param prev_layer_bias: bias from previous layer
        :param debug_mode: bias from previous layer
        :return:
        """
        if prev_layer_output.ndim == 1:
            x = prev_layer_output.reshape(-1, 1)

        if debug_mode:
            print(f'\nIn .forward() of layer: {self.layer_name} :')
            print('BEFORE forward is executed')
            print(f'Prev layer_output_bias.shape:\t{prev_layer_output_bias.shape}')
            print(f'This layer_bias.shape:\t{self._layer_bias.shape}')
            print(f'Prev layer_output.shape:\t{prev_layer_output.shape}')
            print(f'This layer_weights.shape:\t{self._layer_weights.shape}')

        # act(W.X + b), in my implementation I am using act(X.W + b)
        self._layer_output_bias = np.dot(prev_layer_output_bias, self._layer_bias)
        self._layer_output = self.act_fn.execute(
            (np.dot(prev_layer_output, self._layer_weights) + self._layer_output_bias)
        )

        if debug_mode:
            print(f'\nIn .forward() of layer: {self.layer_name} :')
            print('AFTER forward is executed')
            print(f'Prev layer_output_bias.shape:\t{prev_layer_output_bias.shape}')
            print(f'This layer_bias.shape:\t\t\t{self._layer_bias.shape}')
            if self._layer_output_bias is None:
                print('This layer_OUTPUT_bias.shape:\tNone')
            else:
                print(f'This layer_OUTPUT_bias.shape:\t{self._layer_output_bias.shape}')
            print(f'Prev layer_output.shape:\t{prev_layer_output.shape}')
            print(f'This layer_weights.shape:\t{self._layer_weights.shape}')
            if self._layer_output is None:
                print('This layer_OUTPUT.shape:\tNone')
            else:
                print(f'This layer_OUTPUT.shape:\t{self._layer_output.shape}')

        if self._layer_output.ndim == 1:
            self._layer_output = self._layer_output.reshape(1, -1)

    def test(self):
        self.act_fn.execute(888)


class OutputRegression(FullyConnectedLayer):
    layer_type = 'Output_Regression'
    # nodes_in_layer = 1  # hardcode coz regression, the pred is a scaler, hence only one node

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, act_fn, debug_mode=False):
        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer, act_fn, debug_mode)
        self._prediction = 0.0

        if self._debug_mode:
            print('\nOutput Regression Layer <-- __init__ :')
            print(f"self._weights.shape: {self._layer_weights.shape}")
            print(f"self._bias.shape: {self._layer_bias.shape}")
            print(f"self._layer_output.shape: {self._layer_output.shape:}")

    def predict(self, rounding=2):
        """
        regression prediction is linear_activation(wx + b), and this is ALREADY done in the forward
        pass.
        Hence, to 'predict', all we need to do here is to return the layer_matrix that was
        the result of the forward pass.
        :return:
        """
        return np.round(self._layer_output, rounding)


class OutputBinaryClassification(FullyConnectedLayer):
    layer_type = 'Output_Binary_Classification'

    def __init__(self, layer_name, nodes_prev_layer, nodes_in_layer, act_fn):
        # hardcode coz binary classification
        nodes_in_layer = 1
        super().__init__(layer_name, nodes_prev_layer, nodes_in_layer, act_fn)

        self._predicted_class = None

    def get_probability_matrix(self):
        return self._layer_output

    def predict(self, threshold=0.5):
        # returns an array of array, hence [0][0] to get the int

        # both the below are the same, but the .astype() ver is supposed to be twice as fast
        # but ver B can only produce 0 and 1 while ver A is more flexible
        # self._predicted_class = np.where(self._layer_matrix >= threshold, 1, 0)]  # ver A
        self._predicted_class = (self._layer_output > threshold).astype(int)  # ver B
        # nb: .round() is the slowest, in some cases 10 times slower
        # ver A and B are faster than list comprehension coz they are vectorized.
        return self._predicted_class


# not implemented, act fn sh be softmax
# class Output_MultiClass_Classification(Fully_Connected_Layer): # predict 1 class out of multiple classes
# class Output_MultiLabel_Classification(Fully_Connected_Layer): # predict p-values of 1 or more classes
#   an input can belong to >1 class


