from nnLayers import InputLayer
from nnLayers import OutputRegression
from nnActivations import Linear
from nnModel import Model


def perceptron_test():
    #inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2, 3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]
    #targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955]
    inputs = [1, 2, 3, 4]
    targets = [2, 4, 6, 8]

    inputs = [[i] for i in inputs]
    targets = [[i] for i in targets]

    layers = [
        InputLayer('Input', 0, 1),
        OutputRegression('Output', 1, 1, Linear())
    ]

    model = Model(layers)
    model.train(inputs, targets, epochs=20, learning_rate=0.1, print_threshold=5)
    # model.print_model_architecture()
    # errors = model.get_model_error(targets)
    # print(errors, '\n')
    # cost = model.get_model_cost(cost_fn='Mean Absolute Error')
    # print(f"cost: {cost:.02f}")
    # output_layer_weight = layers[-1].get_weights_matrix()
    # print(f"Output_layer_weight: {output_layer_weight}")

    # for inferencing
    preds = model.predict(inputs)
    print(preds, '\n')


if __name__ == '__main__':
    perceptron_test()




