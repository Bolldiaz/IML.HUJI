from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from copy import copy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


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
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:

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


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f)

        # Fit models and predict over training set
        lda = LDA()
        lda.fit(X, y)

        gnb = GaussianNaiveBayes()
        gnb.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy

        models = [lda, gnb]
        models_names = ["LDA", "Gaussian Naive Bayes"]

        symbols = np.array(["circle", "x", 'triangle-up'])
        fig = make_subplots(rows=2, cols=3, subplot_titles=[rf"$\textbf{{{m}}}$" for m in models_names],
                            horizontal_spacing=0.05, vertical_spacing=.1)
        lims = np.array([X.min(axis=0), X.max(axis=0)]).T + np.array([-.4, .4])

        for i, m in enumerate(models):
            acc = np.round(accuracy(y, m._predict(X)), 3)
            fig.add_traces(
                [decision_surface(m._predict, lims[0], lims[1], showscale=False),
                 go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers", showlegend=False,
                            marker=dict(color=y, symbol=symbols[y.astype(int)],
                                        colorscale=['green', 'yellow', 'red'],
                                        line=dict(color="black", width=1)),
                            text=f"accuracy: {acc}",
                            textposition='middle center'),
                 go.Scatter(x=m.mu_[:, 0], y=m.mu_[:, 1], mode='markers',
                            marker=dict(color="black", symbol="x"),
                            showlegend=False),

                 # Add ellipses depicting the covariances of the fitted Gaussians
                 get_ellipse(m.mu_[0], m.cov_ if i == 0 else np.diag(m.vars_[0])),
                 get_ellipse(m.mu_[1], m.cov_ if i == 0 else np.diag(m.vars_[1])),
                 get_ellipse(m.mu_[2], m.cov_ if i == 0 else np.diag(m.vars_[2]))
                 ],
                rows=(i // 2) + 1,
                cols=(i % 2) + 1
            )
            fig.layout.annotations[i].update(text=f'{models_names[i]} accuracy: {acc}')

        fig.update_layout(title=f"Analysis of {f} dataset.", title_x=0.2, height=600, width=1500, )
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
