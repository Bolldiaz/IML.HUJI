from copy import copy

import utils
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from IMLearn.metrics import loss_functions

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """

    data = np.load(f'..//datasets//{filename}')
    return data[:, :2], data[:, 2]


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        perceptron = Perceptron(callback=lambda per: losses.append(per._loss(copy(X), y)))
        perceptron.fit(X, y)

        # Plot figure
        fig = go.Figure([go.Scatter(y=np.asarray(losses), mode='lines')])
        fig.update_xaxes(title_text="iterations")
        fig.update_yaxes(title_text="training loss values")
        fig.update_layout(title_text=f"The perceptron algorithm's training loss values for {n}")
        fig.show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda_acc = loss_functions.accuracy(y, lda.fit(X, y).predict(X))

        gnb = GaussianNaiveBayes()
        gnb_acc = loss_functions.accuracy(y, gnb.fit(X, y).predict(X))
        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy

        models = [lda, gnb]
        models_names = ["LDA", "Gaussian Naive Bayes"]

        symbols = np.array(["circle", "x", 'triangle-up'])
        fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}$" for m in models_names],
                            horizontal_spacing=0.01, vertical_spacing=.03)
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        for i, m in enumerate(models):
            acc = loss_functions.accuracy(y, m.fit(X, y)._predict(X))
            fig.add_traces([utils.decision_surface(m._predict, lims[0], lims[1], showscale=False),
                            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                                       marker=dict(color=y, symbol=symbols[y.astype(int)],
                                                   colorscale=['green', 'yellow', 'red'],
                                                   line=dict(color="black", width=1)),
                                       text=f"accuracy: {acc}",
                                       textposition='middle center'),
                            go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers',
                                       marker=dict(color="black", symbol="x"),
                                       showlegend=False)
                            ],
                           rows=(i // 2) + 1,
                           cols=(i % 2) + 1)

            fig.layout.annotations[i].update(text=f'{models_names[i]} accuracy: {acc}')

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_trace(get_ellipse(lda.mu_[0], lda.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(lda.mu_[1], lda.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(lda.mu_[2], lda.cov_), row=1, col=2, )
        fig.add_trace(get_ellipse(GNB.mu_[0], np.diag(GNB.vars_[0])),
                      row=1, col=1, )
        fig.add_trace(get_ellipse(GNB.mu_[1], np.diag(GNB.vars_[1])),
                      row=1, col=1, )
        fig.add_trace(get_ellipse(GNB.mu_[2], np.diag(GNB.vars_[2])),
                      row=1, col=1, )

        fig.update_layout(title=f"analysis of {f} data set ")
        fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()