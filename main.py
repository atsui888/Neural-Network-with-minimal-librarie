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
from nnLayers import OutputBinaryClassification
from nnLayers import OutputMultiClassClassification
from nnActivations import Linear
from nnActivations import Sigmoid
from nnActivations import Softmax
from nnModel import Model
from nnData_Helper import DataHelper as dh
from nnNormalise import Normalise
from visualise import scatter_plot
from visualise import line_plot
import pickle


def regression_perceptron_input_one_feature():
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
    epochs, cost = model.train(train_targets, epochs=600,
                               learning_rate=0.005,
                               cost_fn='Mean Squared Error',
                               print_threshold=20, debug_mode=DEBUG)
    # line_plot(epochs, cost)

    # for inferencing
    # test_inputs = np.array([1.4, 2.7, 5.0])
    # test_targets = [815, 1200, 1790]

    test_inputs = np.array([4.0]).reshape(-1, 1)  # test input 4, expected results: estd 1,600, beware over-fitting
    preds, probs = model.predict(test_inputs, debug_mode=False)
    print(f"test input: {test_inputs}")
    # print(f"expected prediction: {test_targets}")
    print(f"prediction results:\n{preds}" )
    print('\nIf the results of predicting the unseen inputs are good,')
    print('maybe we should save the current weights of the Network.')


def regression_perceptron_input_two_features():
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
    epochs, cost = model.train(train_targets, epochs=2800,
                               learning_rate=0.2,  # 0.1 - 19,351
                               cost_fn='Mean Squared Error',
                               print_threshold=1000, debug_mode=DEBUG)
    # line_plot(epochs, cost)

    model.save()
    del model

    file_name = 'my_nn_module.pkl'
    with open(file_name, 'rb') as handle:
        model = pickle.load(handle)

    # model.print_model_architecture()

    # for inferencing
    print("\n", "*"*50)
    print("Inferencing:")
    test_inputs = np.array([(1, 10000), (3, 20000), (4, 50000), (5, 20000), (1, 45000)])
    test_inputs_n = normalise.transform(test_inputs)  # do not .fit(), use the min-max already fitted from train set

    # print(f"\nTest Inputs shape: {test_inputs.shape}")
    # print(f"\nTest Inputs:\n{test_inputs_n}")

    test_targets = np.array([500, 850, 1650, 950, 1375]).reshape(-1, 1)

    preds, probs = model.predict(test_inputs_n, debug_mode=False)

    print(f"expected prediction: \n{test_targets}")
    print(f"prediction results: \n{preds[:,0]}")


def classification_perceptron_in_two_features_out_one_node():
    """
    For classification, training data needs to be categorised, output  layer activation function
    needs to be appropriate, e.g. using sigmoid instead of linear.
    Cost function needs to be appropriate, MSE not suitable.
    Test data needs to be categorised e.g. 0 or 1 (if binary classification) and not $1800, $19nn

    :return:
    """
    # ------------------ TRAIN DATA start  --------------------------------
    train_inputs = [(0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000),
                    (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2, 40000), (3.3, 21000), (3.5, 45000),
                    (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]
    train_inputs = np.array(train_inputs)
    if train_inputs.ndim == 1:
        train_inputs = train_inputs.reshape(-1, 1)

    normalise = Normalise(train_inputs)
    normalise.fit().transform()
    train_inputs_n = normalise.get_normalised_matrix()

    train_targets = np.array([230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290, 870, 1545,
                              1480, 1750, 1845, 1790, 1955])
    # convert train targets to two categories
    # 0 = keep, 1 = sell
    train_targets = np.where(train_targets < 1600, 0, 1).reshape(-1, 1)
    # ------------------ train data end  --------------------------------

    # ------------------ DEFINE and INSTANTIATE NETWORK MODEL start  ----
    layers = [
        InputLayer('Input', 0, 2, train_inputs_n, debug_mode=DEBUG),
        OutputBinaryClassification('Output', 2, 1, Sigmoid(), debug_mode=DEBUG)
    ]

    model = Model(layers)
    # ------------------ define and instantiate network model end  ----

    # ------------------ TRAIN NETWORK start  -------------------------
    # MSE + epochs 20 + LR 0.1 gives correct result, cost = 0.0, but is 0 also better?
    # does MSE model the loss better than Log Loss?
    # Log Loss + epochs 20 + LR 0.1 gives correct result, cost = 0.58
    # Log Loss + epochs 400 + LR 0.5 gives correct result, cost = 0.53
    # Log Loss + epochs 120 + LR 0.5 gives correct result, cost = 0.52
    # notes: cost can go down overall yet prediction can be wrong, because the curve fit is usually not exact.
    # cost is an average, i.e. on average we will get better predictions when cost is going towards 0.
    epochs, cost = model.train(train_targets, epochs=800,
                               learning_rate=0.5,
                               cost_fn='Log Loss',  # 'Log Loss', 'Mean Squared Error'
                               print_threshold=100, debug_mode=DEBUG)
    # line_plot(epochs, cost)
    # ------------------ train network end    -------------------------

    # ------------------ SAVE AND LOAD NETWORK start  -------------------------
    # model.save()
    # del model
    #
    # file_name = 'my_nn_module.pkl'
    # with open(file_name, 'rb') as handle:
    #     model = pickle.load(handle)
    # model.print_model_architecture()
    # ------------------ save and load network start  -------------------------

    # ------------------ TEST DATA start  --------------------------------
    print("\n", "*" * 50)
    print("Inferencing:")
    # test inputs
    test_inputs = np.array([(1, 10000), (3, 20000), (4, 50000), (5, 20000), (1, 45000)])
    test_inputs_n = normalise.transform(test_inputs)  # do not .fit(), use the min-max already fitted from train set

    # convert test targets to two categories
    # 0 = keep, 1 = sell
    test_targets = np.array([500, 850, 1650, 950, 1375])
    test_targets = np.where(test_targets < 1600, 0, 1)

    # ------------------ test data end  --------------------------------

    # ------------------ PREDICT start  --------------------------------
    preds, probs = model.predict(test_inputs_n, debug_mode=False)

    print(f"expected prediction: \n{test_targets}")
    print(f"prediction results: \n{preds[:,0]}")
    prob_lst = [round(p, 2) for p in probs[:, 0]]
    print(f"prediction proba: \n{prob_lst}")
    # ------------------ predict end   --------------------------------


def classification_perceptron_in_two_features_out_three_nodes():
    """
    multi-class classification. Instead of using 1 node and cutting segments to represent 3 classes,
    we use 3 nodes - one for each of the 3 classes
    input features (Length, Width), output classes (Red, Green, Blue)
    :return:
    """

    inputs = [(0.0000, 0.0000), (0.2778, 0.2500), (0.2778, 0.9375), (0.9167, 0.6563),
              (0.4167, 0.2500), (0.3611, 0.3438), (0.3333, 0.4063), (0.9722, 0.3750),
              (0.0833, 0.3438), (0.6389, 0.3438), (0.4167, 0.6875), (0.7500, 0.6875),
              (0.0833, 0.1875), (0.9167, 0.5313), (0.1389, 0.2500), (0.8333, 0.6250),
              (0.8056, 0.6250), (0.1944, 1.0000), (0.8333, 0.5625), (0.4167, 1.0000),
              (1.0000, 0.6875), (0.4722, 0.6563), (0.3611, 0.5625), (0.4722, 0.8438),
              (0.1667, 0.3125), (0.4167, 0.9375), (0.3611, 0.9688), (0.9167, 0.3438),
              (0.0833, 0.0313), (0.3333, 0.8750)]
    train_inputs = np.array(inputs)

    red = 0
    green = 1
    blue = 2

    targets = [red, red, blue, green, red, red, red, green, red, green, blue, green, red,
               green, red, green, green, blue, green, blue, green, blue, blue, blue,
               red, blue, blue, green, red, blue]

    train_targets = np.array(targets).reshape(-1, 1)

    # print(f"train_inputs.shape: {train_inputs.shape}")
    # print(f"train_targets.shape: {train_targets.shape}")

    layers = [
        InputLayer('Input', 0, 2, train_inputs, debug_mode=DEBUG),
        OutputMultiClassClassification('Output', 2, 3, Softmax(), debug_mode=DEBUG)
    ]

    model = Model(layers)

    epochs, cost = model.train(train_targets, epochs=50,
                               learning_rate=0.1,
                               cost_fn='Log Loss',  # 'Log Loss', 'Mean Squared Error'
                               print_threshold=5, debug_mode=False)
    # line_plot(epochs, cost)




    # test_inputs = [(0.0278, 0.0313), (0.0556, 0.0625), (0.1111, 0.1563), (0.3611, 0.3750),
    #                (0.2778, 0.3438), (0.8333, 0.3750), (0.5556, 0.4375), (0.8333, 0.5313),
    #                (0.8611, 0.6563), (0.8056, 0.5625), (0.4722, 0.6563), (0.3611, 0.5625),
    #                (0.4722, 0.8438), (0.3611, 0.9688), (0.4167, 0.9375)]
    # test_inputs = np.array(test_inputs)
    #
    # test_targets = [red, red, red, red, red, green, green, green, green, green, blue, blue, blue, blue, blue]


if __name__ == '__main__':
    DEBUG = False

    # -- regression
    # regression_perceptron_input_one_feature()  # target: 1,600
    # regression_perceptron_input_two_features()

    # -- classification
    # classification_perceptron_in_two_features_out_one_node()
    classification_perceptron_in_two_features_out_three_nodes()


