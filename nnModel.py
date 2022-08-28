import numpy as np
from nnData_Helper import DataHelper


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

    def __init__(self, layers):
        self._layers = layers  # a list of layer objects
        self._preds = None
        self._probs = None
        self._errors = None
        self._len_targets = None
        self._cost = None
        self._error_derivative = None
        self._weight_delta = None

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

    def train(self, data, targets, epochs=1, learning_rate=0.001, threshold=0.5,
              print_threshold=10):
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
        for epoch in range(epochs):
            # print(f"epoch #: {epoch + 1}")
            for idx, layer in enumerate(self._layers):
                layer_details = layer.get_layer_details()
                if layer.layer_type.lower() == 'input layer':
                    layer.set_data(data)
                    continue
                else:
                    layer.forward(self._layers[idx - 1].get_layer_matrix())

                # Need to predict, otherwise won't know the error and cannot perform back prop
                # layers (except for input) only does dot pdt in the forward pass, so we need to
                # drive the prediction, calc error, find derivatives, update weights from here.
                if layer_details['name'].lower() == 'output':
                    if layer.layer_type.lower() == 'output_regression':
                        # need to pass the prev layer matrix
                        self._preds = layer.predict()
                    else:
                        self._preds = layer.predict(threshold=threshold)

                    errors = self.get_model_error(targets)
                    cost = self.get_model_cost(cost_fn='Mean Absolute Error')
                    output_layer_weights = layer.get_weights_matrix()

                    if epoch % print_threshold == 0 or epoch == epochs-1:
                        print(f"epoch #{epoch+1}: \t\tWeight: {output_layer_weights} \tCost: {cost:,.02f}")

                    # todo: back prop - update weights (is this the best place to do this?
                    layer.update_weights_matrix(learning_rate, cost)

                self._probs = layer.get_layer_matrix()

    def predict(self, data, threshold=0.5):
        """
        This fn is used for inferencing, NOT for training

        For the purpose of backprop, the model.forward() already does a .predict() there
        so that we can calc the errors, which is used for back prop.

        :param data:
        :param threshold:
        :return:
        """
        # Forward Pass ONCE and does a prediction
        preds = None

        for idx, layer in enumerate(self._layers):
            # print(f"\nlayer.layer_type: {layer.layer_type}")
            if layer.layer_type.lower() == 'input layer':
                layer.set_data(data)
                continue
            else:
                # Forward Once
                layer.forward(self._layers[idx - 1].get_layer_matrix())

                # print(f"after forward pass:\n{layer.get_layer_matrix()}")

            if layer.layer_name.lower() == 'output':
                if layer.layer_type.lower() == 'output_regression':
                    print('\nPrediction: Regression')
                    self._preds = layer.predict()
                    preds = [f"{p[0]:,.02f}" for p in self._preds]
                else:
                    print('classification')
                    self._preds = layer.predict(threshold=threshold)
                    preds = 'do something here'

                self._probs = layer.get_layer_matrix()
                return preds

    def get_proba(self):
        return self._probs

    def get_model_error(self, targets):
        # model's difference between predictions and targets

        if DataHelper.is_list_of_lists(targets):
            # print('list of lists targets ok')
            self._len_targets = len(targets)
            # errors = preds - targets
            # preds = weights * input
            # so, errors = (weights * input) - targets #
            # targets and inputs are given, we can only adjust the weights to reduce the error
            if self._preds is None:
                raise ValueError('Preds cannot be None.')
            self._errors = self._preds - targets  # a matrix
            return self._errors

    def get_model_cost(self, cost_fn='Mean Squared Error'):
        # sum(errors) / len(targets) to obtain a single scalar
        # there are many types of cost fns e.g.
        # For regression: "Mean Absolute Error", "Mean Squared Error"
        # For classification: "Binary Cross Entrophy", "Categorical Cross Entrophy"
        if self._errors is None:
            raise ValueError('model errors is None. \nYou must call model.get_model_error(targets) first.')

        if cost_fn == 'Mean Squared Error':
            self._cost = np.sum(np.square(self._errors)) / self._len_targets
        elif cost_fn == 'Mean Absolute Error':
            self._cost = np.sum(np.abs(self._errors)) / self._len_targets
        return self._cost

    def get_weight_delta(self):
        self._error_derivative = 2 * self._errors

        # self._weight_delta = self._error_derivative * Inputs
        # rc: inputs is the input layer or the preceding layer ?

        return self._weight_delta

    def update_weights(self):
        # weights -= learning_rate * np.sum(self._weight_delta) / len(self._weight_delta)
        pass

    def save_model_architecture(self):
        pass

    def save_trained_model(self):
        pass

    def load_model_architecture(self):
        pass

    def load_trained_model(self):
        pass
