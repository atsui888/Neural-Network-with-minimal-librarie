"""
Software Dev: richard chai
https://www.linkedin.com/in/richardchai/

todo: add normalize method to input layer class

useful resources:
https://machinelearningmastery.com/learning-rate-for-deep-learning-neural-networks/#:~:text=A%20traditional%20default%20value%20for,starting%20point%20on%20your%20problem.
- Learning Rate
--- Grid Search
--- Adding Momentum, Decay
--- Learning Rate Schedule
--- Adaptive Learning Rates
- http://blog.dlib.net/2017/12/a-global-optimization-algorithm-worth.html
- https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/
"""
import numpy as np
from nnLayers import InputLayer
from nnLayers import OutputRegression
from nnActivations import Linear
from nnModel import Model
from nnData_Helper import DataHelper as dh
from nnNormalise import Normalise
from visualise import scatter_plot
from visualise import line_plot


def perceptron_test_input_one_feature():
    """
    inputs are very small compared to targets e.g. 0.2 --> 230, 2.0 --> 1140
    target = W.x
    if x is tiny, W needs to be HUGE! This will lead to exploding gradients and this problem will be
    worse in a deep network. One way to mitigate would be to use extremely tiny learning rates, maybe
    it might be better to scale the targets.
    See perceptron_test_input_one_feature_type_b()

    :return:
    """

    train_inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2, 3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]

    train_inputs = np.array(train_inputs)
    if train_inputs.ndim == 1:
        train_inputs = train_inputs.reshape(-1, 1)
    if DEBUG:
        print('\n"perceptron_test_input_one_feature_type_a"')
        print("Initial Data Preparation (before training loop)")
        print("-" * 50)
        print(f"type(train_inputs): {type(train_inputs)}")
        print(f"train_inputs.shape: {train_inputs.shape}")
        print(f"consisting of {train_inputs.shape[0]} samples of {train_inputs.shape[1]} features each.\n")

    train_targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545,
                     1480, 1750, 1845, 1790, 1955]
    train_targets = np.array(train_targets).reshape(-1, 1)
    if DEBUG:
        print(f"type(train_targets): {type(train_targets)}")
        print(f"train_targets.shape: {train_targets.shape}")
        print(f"consisting of {train_targets.shape[0]} samples of {train_targets.shape[1]} target each.\n")

    # scatter_plot(train_inputs, train_targets)

    layers = [
        InputLayer('Input', 0, 1, train_inputs, debug_mode=DEBUG),
        OutputRegression('Output', 1, 1, Linear(), debug_mode=DEBUG)
    ]

    model = Model(layers)
    epochs, cost = model.train(train_targets, epochs=25,
                               learning_rate=0.02,
                               cost_fn='Mean Squared Error',
                               print_threshold=100, debug_mode=DEBUG)
    # line_plot(epochs, cost)

    # for inferencing
    # test_inputs = np.array([1.4, 2.7, 5.0])
    # test_targets = [815, 1200, 1790]

    test_inputs = np.array([4.0]).reshape(-1, 1)
    preds, probs = model.predict(test_inputs, debug_mode=False)
    print(f"test input: {test_inputs}")
    # print(f"expected prediction: {test_targets}")
    print(f"prediction results:\n{preds}" )
    print('\nIf the results of predicting the unseen inputs are good,')
    print('maybe we should save the current weights of the Network.')


def perceptron_test_input_two_features():
    # train_inputs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]  # 2 features, 6 data samples
    # train_targets = [12, 14, 16, 18, 20, 22]

    train_inputs = [(0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000),
                    (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2, 40000), (3.3, 21000), (3.5, 45000),
                    (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]
    train_inputs = np.array(train_inputs)
    if train_inputs.ndim == 1:
        train_inputs = train_inputs.reshape(-1, 1)

    normalise = Normalise(train_inputs)
    normalise.fit().transform()
    train_inputs_n = normalise.get_normalised_matrix()

    train_targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545,
                     1480, 1750, 1845, 1790, 1955]
    train_targets = np.array(train_targets)
    train_targets = np.array(train_targets).reshape(-1, 1)

    if DEBUG:
        print('\n"perceptron_test_input_two_features"')
        print("Initial Data Preparation (before training loop)")
        print("-"*50)
        print(f"type(train_inputs_norm): {type(train_inputs_n)}")
        print(f"train_inputs_norm.shape: {train_inputs_n.shape}")
        print(f"consisting of {train_inputs_n.shape[0]} samples of {train_inputs_n.shape[1]} features each.\n")

    if DEBUG:
        print(f"type(train_targets): {type(train_targets)}")
        print(f"train_targets.shape: {train_targets.shape}")
        print(f"consisting of {train_targets.shape[0]} samples of {train_targets.shape[1]} target each.\n")

    # scatter_plot(train_inputs_n, train_targets)  # this plot is only 2d, will fail for train_inputs_n

    layers = [
        InputLayer('Input', 0, 2, train_inputs_n, debug_mode=DEBUG),
        OutputRegression('Output', 2, 1, Linear(), debug_mode=DEBUG)
    ]

    model = Model(layers)
    epochs, cost = model.train(train_targets, epochs=4000,
                               learning_rate=0.15,  # 0.1 - 19,351
                               cost_fn='Mean Squared Error',
                               print_threshold=10000, debug_mode=DEBUG)
    # line_plot(epochs, cost)

    # for inferencing
    print("\n", "*"*50)
    print("Inferencing:")
    test_inputs = np.array([(1, 10000), (3, 20000), (4, 50000), (5, 20000), (1, 45000)])
    test_inputs_n = normalise.transform(test_inputs)  # do not .fit(), use the min-max obtained from train set

    # print(f"\nTest Inputs shape: {test_inputs.shape}")
    print(f"\nTest Inputs:\n{test_inputs_n}")

    test_targets = np.array([500, 850, 1650, 950, 1375]).reshape(-1, 1)

    preds, probs = model.predict(test_inputs_n, debug_mode=False)

    print(f"expected prediction: \n{test_targets}")
    print(f"\nprediction results: \n{preds[:,0]}")



if __name__ == '__main__':
    DEBUG = False
    # perceptron_test_input_one_feature()
    perceptron_test_input_two_features()



