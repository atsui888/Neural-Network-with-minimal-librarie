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
from visualise import scatter_plot
from visualise import line_plot


def perceptron_test_input_one_feature_type_a():
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
        OutputRegression('Output', 1, 1, Linear())
    ]

    model = Model(layers)
    epochs, cost = model.train(train_targets, epochs=1,  # 10000
                               learning_rate=0.0000755,  # 0.000075 / 25,818
                               cost_fn='Mean Squared Error',
                               print_threshold=1, debug_mode=DEBUG)
    line_plot(epochs, cost)

    # for inferencing
    test_inputs = [1.4, 2.7, 5.0]
    test_targets = [815, 1200, 1790]
    preds = model.predict(test_inputs)
    print(f"test input: {test_inputs}")
    print(f"expected prediction: {test_targets}")
    print(f"prediction results: {preds}" )
    print('\nIf the results of predicting the unseen inputs are good,')
    print('maybe we should save the current weights of the Network.')


def perceptron_test_input_one_feature_type_b():
    """
    this version is the same as type_a() except that the targets have been scaled (divided by 1,000)
    :return:
    """
    train_inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2, 3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]

    train_inputs = np.array(train_inputs)
    if train_inputs.ndim == 1:
        train_inputs = train_inputs.reshape(-1, 1)
    if DEBUG:
        print('\n"perceptron_test_input_one_feature_type_b"')
        print("Initial Data Preparation (before training loop)")
        print("-" * 50)
        print(f"type(train_inputs): {type(train_inputs)}")
        print(f"train_inputs.shape: {train_inputs.shape}")
        print(f"consisting of {train_inputs.shape[0]} samples of {train_inputs.shape[1]} features each.\n")

    train_targets = [0.230, 0.555, 0.815, 0.860, 1.140, 1.085, 1.200, 1.330, 1.290, 0.870, 1.545,
                     1.480, 1.750, 1.845, 1.790, 1.955]
    train_targets = np.array(train_targets).reshape(-1, 1)
    if DEBUG:
        print(f"type(train_targets): {type(train_targets)}")
        print(f"train_targets.shape: {train_targets.shape}")
        print(f"consisting of {train_targets.shape[0]} samples of {train_targets.shape[1]} target each.\n")

    # scatter_plot(train_inputs, train_targets)

    layers = [
        InputLayer('Input', 0, 1, train_inputs, debug_mode=DEBUG),
        OutputRegression('Output', 1, 1, Linear())
    ]

    model = Model(layers)
    epochs, cost = model.train(train_targets, epochs=1,
                               learning_rate=0.00001,
                               cost_fn='Mean Squared Error',
                               print_threshold=1, debug_mode=DEBUG)
    # line_plot(epochs, cost)

    # for inferencing
    test_inputs = [4.4, 5.0]
    test_targets = [1.845, 1.790]
    preds = model.predict(test_inputs)
    print(f"test input: {test_inputs}")
    print(f"expected prediction: {test_targets}")
    print(f"prediction results: {preds}" )
    print('\nIf the results of predicting the unseen inputs are good,')
    print('maybe we should save the current weights of the Network.')


def perceptron_test_input_two_features():
    # inputs = [0.2, 1.0, 1.4, 1.6, 2.0, 2.2, 2.7, 2.8, 3.2, 3.3, 3.5, 3.7, 4.0, 4.4, 5.0, 5.2]
    # targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545, 1480, 1750, 1845, 1790, 1955]
    # train_inputs = [1, 2, 3, 4, 5]  # original
    train_inputs = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)]  # 2 features, 5 data samples
    train_inputs = np.array(train_inputs)
    if train_inputs.ndim == 1:
        train_inputs = train_inputs.reshape(-1, 1)
    if DEBUG:
        print('\n"perceptron_test_input_two_features"')
        print("Initial Data Preparation (before training loop)")
        print("-"*50)
        print(f"type(train_inputs): {type(train_inputs)}")
        print(f"train_inputs.shape: {train_inputs.shape}")
        print(f"consisting of {train_inputs.shape[0]} samples of {train_inputs.shape[1]} features each.\n")

    train_targets = [12, 14, 16, 18, 20]
    train_targets = np.array(train_targets).reshape(-1, 1)
    if DEBUG:
        print(f"type(train_targets): {type(train_targets)}")
        print(f"train_targets.shape: {train_targets.shape}")
        print(f"consisting of {train_targets.shape[0]} samples of {train_targets.shape[1]} target each.\n")

    # scatter_plot(train_inputs, train_targets)

    layers = [
        InputLayer('Input', 0, 2, train_inputs, debug_mode=DEBUG),
        OutputRegression('Output', 2, 1, Linear(), debug_mode=DEBUG)
    ]

    model = Model(layers)
    epochs, cost = model.train(train_targets, epochs=100,  # 100
                               learning_rate=0.01,
                               cost_fn='Mean Squared Error',
                               print_threshold=10, debug_mode=DEBUG)
    # line_plot(epochs, cost)

    # for inferencing
    test_inputs = np.array([(4, 4), (3, 3)])
    if test_inputs.ndim == 1:
        test_inputs = test_inputs.reshape(-1, 1)
    print(f"\nTest Inputs shape: {test_inputs.shape}")
    test_targets = np.array([18, 16])
    if test_targets.ndim == 1:
        test_targets = test_targets.reshape(-1, 1)
    print(f"Test Targets shape: {test_targets.shape}\n")

    preds = model.predict(test_inputs)
    print(f"test input: {test_inputs}")
    print(f"expected prediction: {test_targets}")
    print(f"prediction results: {preds}" )
    # print('\nIf the results of predicting the unseen inputs are good,')
    # print('maybe we should save the current weights of the Network.')


if __name__ == '__main__':
    DEBUG = False
    # perceptron_test_input_one_feature_type_a()
    # perceptron_test_input_one_feature_type_b()
    perceptron_test_input_two_features()



