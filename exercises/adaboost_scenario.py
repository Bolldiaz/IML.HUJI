import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def decision_boundary(partial_predict, T, xrange, yrange):
    xrange, yrange = np.linspace(*xrange, 120), np.linspace(*yrange, 120)
    xx, yy = np.meshgrid(xrange, yrange)
    pred = partial_predict(np.c_[xx.ravel(), yy.ravel()], T)
    return go.Contour(x=xrange, y=yrange, z=pred.reshape(xx.shape), colorscale=custom, reversescale=False,
                      opacity=.7, connectgaps=True, hoverinfo="skip", showlegend=False, showscale=False)


def fit_and_evaluate_adaboost(noise, n_learners=25, train_size=500, test_size=50):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    training_error, test_error = [], []
    adaBoost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    n_learners_list = np.arange(1, n_learners)
    for T in n_learners_list:
        train_loss, test_loss = adaBoost.partial_loss(train_X, train_y, T), adaBoost.partial_loss(test_X, test_y, T)
        print(f"train: {train_loss}, test: {test_loss}")
        training_error.append(train_loss)
        test_error.append(test_loss)

    fig = go.Figure([go.Scatter(x=n_learners_list, y=training_error, mode='lines', name=r'$Training-Error$'),
                     go.Scatter(x=n_learners_list, y=test_error, mode='lines', name=r'$Test-Error$')])
    fig.update_xaxes(title_text="learners num")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text='Adaboost error')
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{t} iterations}}$" for t in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    for i, T in enumerate(T):  # todo check decision boundary
        fig.add_traces([decision_boundary(adaBoost.partial_predict, T, lims[0], lims[1]),
                        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=train_y, symbol=np.array(["circle", "x"])[train_y],
                                               colorscale=[custom[0], custom[-1]], line=dict(color="black", width=1)))],
                       rows=(i // 3) + 1,
                       cols=(i % 3) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision boundary by using changed ensemble}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)

    # Question 3: Decision surface of best performing ensemble
    raise NotImplementedError()

    # Question 4: Decision surface with weighted samples
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
