import time
import numpy as np
import gzip
from typing import Tuple, Callable, List

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, CrossEntropyLoss, softmax, Id
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from exercises.gradient_descent_investigation import get_gd_state_recorder_callback

pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1, 2).reshape(height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray")\
        .update_layout(title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
                       font=dict(size=16), coloraxis_showscale=False)\
        .update_xaxes(showticklabels=False)\
        .update_yaxes(showticklabels=False)


def sgd_time_record(y) -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    times = []
    losses = []

    def callback(solver, weights, val, grad, t, eta, delta, batch_indices=None) -> None:
        times.append(time.time())
        losses.append(np.mean(val))

    return callback, times, losses


def gd_time_record(y) -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    times, losses = [], []

    def callback(solver, weights, val, grad, t, eta, delta):
        times.append(time.time())
        losses.append(np.mean(val))

    return callback, times, losses


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    callback, values, grads, weights = get_gd_state_recorder_callback()

    nn_1 = NeuralNetwork(
        modules=[FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=Id(),
                                     include_intercept=True)],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(max_iter=10000, learning_rate=FixedLR(1e-1), callback=callback,
                                         batch_size=256))

    # used to save time (instead of retraining, use the weights from the previous run)
    isSaved = False
    import pickle

    # check if network is saved in a file
    if not isSaved:
        nn_1.fit(train_X, train_y)

        #  save network to pickle
        with open('nn_1.pkl', 'wb') as f:
            pickle.dump(nn_1.weights, f)
    else:
        with open('nn_1.pkl', 'rb') as f:
            nn_1.weights = pickle.load(f)

    nn_1.fit(train_X, train_y)
    print(accuracy(test_y, nn_1._predict(test_X)))

    # Plotting convergence process
    fig = go.Figure(data=[go.Scatter(x=list(range(len(values))), y=[np.sum(value) for value in values])],
                    layout=go.Layout(title=r"$\text{Objective Function Convergence}$",
                                     xaxis=dict(title=r"$\text{Iteration}$"),
                                     yaxis=dict(title=r"$\text{Objective}$")))
    # add norm of weights
    fig.add_trace(go.Scatter(x=list(range(len(grads))), y=[np.linalg.norm(grad) for grad in grads]))
    fig.show()

    # Plotting test true- vs predicted confusion matrix
    conf_mat = confusion_matrix(test_y, nn_1._predict(test_X))
    fig = go.Figure(data=[go.Heatmap(z=conf_mat,
                                     x=list(range(n_classes)),
                                     y=list(range(n_classes)))],
                    layout=go.Layout(title=r"$\text{Confusion Matrix}$",
                                     xaxis=dict(title=r"$\text{True}$"),
                                     yaxis=dict(title=r"$\text{Predicted}$")))
    fig.show()

    # print(np.unravel_index(np.argsort(conf_mat, axis=None)[:3], conf_mat.shape))
    print("3 least confused:")
    for i, j in zip(*np.unravel_index(np.argsort(conf_mat, axis=None)[:3], conf_mat.shape)):
        print(f"{i} confused with {j}")

    np.fill_diagonal(conf_mat, 0)
    # print(np.unravel_index(np.argsort(conf_mat, axis=None)[-2:], conf_mat.shape))
    print("2 most confused:")
    for i, j in zip(*np.unravel_index(np.argsort(conf_mat, axis=None)[-2:], conf_mat.shape)):
        print(f"{i} confused with {j}")

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    callback, values, grads, weights = get_gd_state_recorder_callback()
    nn_2 = NeuralNetwork(
        modules=[FullyConnectedLayer(input_dim=n_features, output_dim=n_classes, activation=Id(),
                                     include_intercept=True)],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(max_iter=10000, learning_rate=FixedLR(1e-1), callback=callback,
                                         batch_size=256))

    nn_2.fit(train_X, train_y)
    print(accuracy(test_y, nn_2._predict(test_X)))

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    # leave only test_X where y = 7
    test_X = test_X[test_y == 7]
    test_y = test_y[test_y == 7]

    # get 64 most confident samples
    confidences = nn_1.compute_prediction(test_X)
    confidences = np.apply_along_axis(lambda x: np.max(x), 1, confidences)

    # plot 64 most confident samples
    im = plot_images_grid(test_X[np.argsort(confidences)[-64:]].reshape(64, 784), title="Most confident")
    im.show()
    im = plot_images_grid(test_X[np.argsort(confidences)[:64]].reshape(64, 784), title="Least confident")
    im.show()

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#

    train_X = train_X[:2500]
    train_y = train_y[:2500]

    gd_callback, gd_times, gd_losses = gd_time_record(train_y)

    gd_nn = NeuralNetwork(
        modules=[FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=Id(),
                                     include_intercept=True)],
        loss_fn=CrossEntropyLoss(),
        solver=GradientDescent(max_iter=5000, learning_rate=FixedLR(1e-1), callback=gd_callback, tol=1e-10))

    gd_nn.fit(train_X, train_y)

    # subtract gd_times[0] from gd_times to get running time of gd
    gd_times = np.array(gd_times) - gd_times[0]
    gd_losses = np.array(gd_losses)

    # plot loss as function of runtime
    fig = go.Figure(data=[go.Scatter(x=gd_times, y=gd_losses)],
                    layout=go.Layout(title=r"$\text{Loss Function Convergence}$",
                                     xaxis=dict(title=r"$\text{Runtime}$"),
                                     yaxis=dict(title=r"$\text{Loss}$")))
    fig.show()

    # ---------------------------------------------------------------------------------------------#
    sgd_callback, sgd_times, sgd_losses = sgd_time_record(train_y)

    sgd_nn = NeuralNetwork(
        modules=[FullyConnectedLayer(input_dim=n_features, output_dim=64, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=64, output_dim=64, activation=ReLU(),
                                     include_intercept=True),
                 FullyConnectedLayer(input_dim=64, output_dim=n_classes, activation=Id(),
                                     include_intercept=True)],
        loss_fn=CrossEntropyLoss(),
        solver=StochasticGradientDescent(max_iter=5000, learning_rate=FixedLR(1e-1), callback=sgd_callback,
                                         batch_size=64, tol=1e-10))

    sgd_nn.fit(train_X, train_y)

    # subtract sgd_times[0] from sgd_times to get running time of sgd
    sgd_times = np.array(sgd_times) - sgd_times[0]
    sgd_losses = np.array(sgd_losses)

    # plot loss as function of runtime
    fig = go.Figure(data=[go.Scatter(x=sgd_times, y=sgd_losses)],
                    layout=go.Layout(title=r"$\text{Loss Function Convergence}$",
                                     xaxis=dict(title=r"$\text{Runtime}$"),
                                     yaxis=dict(title=r"$\text{Loss}$")))
    fig.show()

    # plot loss as function of runtime
    fig = go.Figure(data=[go.Scatter(x=gd_times, y=gd_losses), go.Scatter(x=sgd_times, y=sgd_losses)],
                    layout=go.Layout(title=r"$\text{Loss Function Convergence}$",
                                     xaxis=dict(title=r"$\text{Runtime}$"),
                                     yaxis=dict(title=r"$\text{Loss}$")))

    # add sgd to fig
    fig.add_trace(go.Scatter(x=sgd_times, y=sgd_losses, name='SGD'))
    fig.show()
