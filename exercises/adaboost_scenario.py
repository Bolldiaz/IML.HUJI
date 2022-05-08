import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
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


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    training_error, test_error = [], []
    adaBoost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    n_learners_list = np.arange(1, n_learners)
    for T in n_learners_list:
        train_loss, test_loss = adaBoost.partial_loss(train_X, train_y, T), adaBoost.partial_loss(test_X, test_y, T)
        training_error.append(train_loss)
        test_error.append(test_loss)

    fig = go.Figure([go.Scatter(x=n_learners_list, y=training_error, mode='lines', name=r'$Training-Error$'),
                     go.Scatter(x=n_learners_list, y=test_error, mode='lines', name=r'$Test-Error$')])
    fig.update_xaxes(title_text="learners num")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text=rf"$\textbf{{Adaboost error}}$")
    fig.write_image(f"ex4/AdaBoost_Train&TestErrors_Noise{noise}.png")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"{i} learners" for i in T],
                        horizontal_spacing=0.05, vertical_spacing=0.05
                        )

    m = go.Scatter(x=test_X[:, 0],
                   y=test_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   name="Label 1",
                   marker=dict(color=(test_y == 1).astype(int),
                               symbol=class_symbols[test_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black",
                                         width=1)))
    for i, t in enumerate(T):
        fig.add_traces(
            [
                decision_surface(lambda x: adaBoost.partial_predict(x, t),
                                 lims[0],
                                 lims[1],
                                 showscale=False), m
            ],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        width=800,
        height=900,
        title=rf"$\textbf{{Decision Boundaries Of AdaBoost based on number of learners: noise={noise}}}$",
        margin=dict(t=100)
    )
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)

    fig.write_image(f"ex4/AdaBoost_DecisionBoundaries_Noise{noise}.png")
    # fig.show()

    # Question 3: Decision surface of best performing ensemble
    t_min = np.argmin(test_error) + 1
    acc = np.round(1-test_error[t_min], 3)
    fig = go.Figure([decision_surface(lambda x: adaBoost.partial_predict(x, t_min), lims[0], lims[1], showscale=False), m])
    fig.update_xaxes(matches='x', range=[-1, 1], constrain="domain")
    fig.update_yaxes(matches='y', range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(title_text=f"Ensemble achieved the lowest test error\nsize={t_min}, accuracy={acc}")

    fig.write_image(f"ex4/AdaBoost_LowestBoundary_Noise{noise}.png")
    # fig.show()

    # Question 4: Decision surface with weighted samples
    m = go.Scatter(x=train_X[:, 0],
                   y=train_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=(train_y == 1).astype(int),
                               size=adaBoost.D_ / np.max(adaBoost.D_) * 5,
                               symbol=class_symbols[train_y.astype(int)],
                               colorscale=[custom[0], custom[-1]],
                               line=dict(color="black", width=1))
                   )

    fig = go.Figure([decision_surface(adaBoost.predict, lims[0], lims[1], showscale=False), m])
    fig.update_xaxes(range=[-1, 1], constrain="domain")
    fig.update_yaxes(range=[-1, 1], constrain="domain", scaleanchor="x", scaleratio=1)
    fig.update_layout(dict1=dict(width=600,
                                 height=600,
                                 title=rf"$\textbf{{Adaboost train set with sample sized by weights, noise={noise}}}$"))

    fig.write_image(f"ex4/AdaBoost_Weighted_Samples_Noise{noise}.png")
    # fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)
    fit_and_evaluate_adaboost(noise=0.4)
