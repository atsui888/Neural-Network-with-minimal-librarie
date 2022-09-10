import numpy as np
import pickle


class Model:
    """
        Model Error depends on ALL layers, not just the output layer,
        hence, errors should be calc at this level, not at the output layer level.

        contains history of each epoc's weights, preds and cost
        hence, for each epoc:
            `layer class` takes forwards and predicts
            preds are stored in this `class Model`
            costs is caculated and stored here
            weights are stored here too.

        and
        def .save_model_architecture(), saves to a pickle:
            . the model (i.e. all layers and their parameters)
        def .save_trained_model()
            . saves the weights and bias
        def .load_model_architecture()
        def .load_trained_model()
            load weights and bias and inits it to the right layer

    """

    def __init__(self, layers=None):
        if layers is not None:
            self._layers = layers  # a list of layer objects
        self._preds = None
        self._probs = None
        self._errors = None
        self._len_targets = None
        self._cost = None
        self._error_derivatives = None
        self._weight_deltas = None
        self._bias_deltas = None

    def print_model_architecture(self):
        for idx, layer in enumerate(self._layers):
            print(f"\nLayer #{idx}:")
            layer.print_layer_details()

        return True

    def matrix_mul_dims_ok(self, layer_prev, weights):
        # class method
        m, n1 = layer_prev.shape
        print(f'\n\nLprev.shape: \tm = {m} \tby n = {n1}')

        n2, p = weights.shape
        print(f'x.shape: \tn = {n2} \tby p = {p}')

        if n1 != n2:
            print('n1 != n2, matrix multiplication will fail.\n')
            print('either A or B should be reshaped if possible.')
            return False
        else:
            return True

    def train(self, targets, epochs=1, learning_rate=0.001, cost_fn='Mean Squared Error',
              threshold=0.5, print_threshold=10, debug_mode=False):
        """
        X: train data
        y: targets
        epochs: Number of times we loop through a complete dataset
                consisting of N rows.

        each epoch, we save a history of:
        . weights, bias, preds, costs

        for each epoch:
            forward once
            get preds
            calc error
            calc cost
            back prop (update weights)

        """
        epoch_lst = []
        cost_lst = []

        for epoch in range(epochs):
            # print(f"epoch #: {epoch + 1}")
            for idx, layer in enumerate(self._layers):
                layer_details = layer.get_layer_details()
                if layer.layer_type.lower() == 'input layer':
                    # print(f"Input Layer shape --> {layer.get_layer_matrix().shape}")
                    continue
                else:
                    layer.forward(self._layers[idx - 1].get_layer_output(),
                                  self._layers[idx - 1].get_layer_output_bias(),
                                  debug_mode=debug_mode)

                # PREDICT
                if layer_details['name'].lower() == 'output':
                    if layer.layer_type.lower() == 'output_regression':
                        # need to pass the prev layer matrix
                        self._preds = layer.predict()
                    else:
                        self._preds = layer.predict(threshold=threshold)

                    _ = self.get_model_error(targets)

                    cost = self.get_model_cost(cost_fn=cost_fn)

                    epoch_lst.append(epoch)
                    cost_lst.append(cost)

                    if epoch % print_threshold == 0 or epoch == epochs-1:
                        print(f"epoch #{epoch+1}: \tW.shape: {layer.get_weights_matrix().shape},\tB.shape: {layer.get_bias_matrix().shape}, \tCost: {cost:,.02f}")

                    # get_model_error() already done above
                    # Get Weight_Derivatives
                    # .calc_weight_deltas() is part of model class but the param
                    # is the layer's matrix. Thus, .calc_weight_deltas() is actually
                    # using the layer's data/matrix to run calculations and then
                    # calling the layer's `update weights` method so that the layer
                    # updates its own weights. Similar concept for bias.
                    self.calc_weight_bias_deltas(
                        self._layers[idx-1].get_layer_output(),
                        self._layers[idx].get_layer_output_bias()
                    )

                    # todo: for perceptron this works, but what if there is >=1 hidden layer?
                    # todo: keep this or change?
                    # todo: back prop - update weights (is this the best place to do this?
                    # updating the weights in the network is called `back propagation`
                    # `back propagation` takes the desired weight changes and propagates it back to
                    # the start of the network by adjusting the weights
                    layer.update_weights_bias(learning_rate, self._weight_deltas, self._bias_deltas)

                self._probs = layer.get_layer_output()
        return epoch_lst, cost_lst

    def get_model_error(self, targets):
        # model's difference between predictions and targets
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)

        self._len_targets = len(targets)

        if self._preds is None:
            raise ValueError('Preds cannot be None.')

        self._errors = self._preds - targets  # a matrix

        return self._errors

    def get_model_cost(self, cost_fn='Mean Squared Error'):
        # Cost - a metric to show how far off we are from the target
        #      aka Average Error
        # usually not used in back prop
        # sum(errors) / len(targets) to obtain a single scalar
        # there are many types of cost fns e.g.

        # For regression <-- "Mean Absolute Error", "Mean Squared Error"
        # For classification <-- "Binary Cross Entropy", "Categorical Cross Entropy"
        if self._errors is None:
            raise ValueError('model errors is None. \nYou must call model.get_model_error(targets) first.')

        if cost_fn == 'Mean Squared Error':
            self._cost = np.sum(np.square(self._errors)) / self._len_targets
        elif cost_fn == 'Mean Absolute Error':
            self._cost = np.sum(np.abs(self._errors)) / self._len_targets
        return self._cost

    def calc_weight_bias_deltas(self, layer_matrix, bias_matrix):
        if self._errors is None:
            raise ValueError('To calculate error derivative, Errors cannot be None ')
        # first, find the error derivatives
        self._error_derivatives = 2 * self._errors
        # next, get weight deltas
        # todo: is weight_date, err_d * input_data
        # todo: OR
        # todo: is weight_date, err_d * input_from_prev_layer
        # weight_derivative is the change that each training sample wants to make to the weight

        # shd be element multiply, not do product
        # https://numpy.org/doc/stable/reference/generated/numpy.multiply.html
        self._weight_deltas = np.sum(
            np.multiply(self._error_derivatives, layer_matrix),
            axis=0) / layer_matrix.shape[0]
        self._bias_deltas = np.sum(
            np.multiply(self._error_derivatives, bias_matrix),
            axis=0) / bias_matrix.shape[0]

        # print(f"self._weight_deltas: \n{self._weight_deltas}")

    def save(self, file_name='my_nn_module.pkl'):
        """
        save entire model
        :param file_name:
        :return:
        """
        with open(file_name, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('model saved')

    def save_model_architecture(self):
        pass

    def save_trained_model(self):
        pass

    def load_model_architecture(self):
        pass

    def load_trained_model(self):
        pass

    def predict(self, input_data, threshold=0.5, debug_mode=False):
        """
        THIS FUNCTION IS USED FOR "INFERENCING", not for training
        only accepts a row vector or a list of row vectors as input where
        each row is a data sample of n number of features where n>=1
        and data is a numpy array.

        :param input_data:
        :param threshold:
        :param debug_mode:
        :return:
        """

        # input data
        if input_data.ndim == 1:
            input_data = input_data.reshape(-1, 1)

        # load model network, weights and bias

        # predict
        for idx, layer in enumerate(self._layers):
            layer_details = layer.get_layer_details()
            if layer.layer_type.lower() == 'input layer':
                layer.set_data_and_bias(input_data)
                continue
            else:
                layer.forward(self._layers[idx - 1].get_layer_output(),
                              self._layers[idx - 1].get_layer_output_bias(),
                              debug_mode=debug_mode)

            # PREDICT
            if layer_details['name'].lower() == 'output':
                if layer.layer_type.lower() == 'output_regression':
                    self._preds = layer.predict()
                else:
                    self._preds = layer.predict(threshold=threshold)

            self._probs = layer.get_layer_output()
        return self._preds, self._probs

    def get_proba(self):
        return self._probs
