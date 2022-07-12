import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics.loss_functions import misclassification_error as MCE
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from sklearn.metrics import roc_curve, auc

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    weights, vals = [], []

    def callback(weight: np.ndarray, val: np.ndarray) -> None:
        vals.append(val)
        weights.append(weight)

    return callback, weights, vals


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    for eta in etas:
        for module, L in [(L1, "L1"), (L2, "L2")]:
            callback, weights, vals = get_gd_state_recorder_callback()

            GradientDescent(learning_rate=FixedLR(eta), callback=callback).fit(module(init), X=None, y=None)
            fig = plot_descent_path(module, np.asarray(weights), title=f"with module: {L} and LR: {eta}")
            # fig.show()

            if eta == 0.01:
                fig = go.Figure([go.Scatter(y=vals, mode='markers')])
                fig.update_layout(
                    xaxis_title="iterations", yaxis_title="convergence",
                    title=f"Convergence-Rate Of module {L} With learning Rate Of {eta}"
                )
                # fig.show()
                print(f"The lowest loss over the convergence of {L}: {np.min(vals)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    plots = []
    for g in gammas:
        callback, weights, vals = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=ExponentialLR(eta, g), callback=callback).fit(L1(init), X=None, y=None)
        plots.append(go.Scatter(y=vals, mode='markers+lines', name=f"gamma={g}"))

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure(plots)
    fig.update_layout(
        xaxis_title="iteration", yaxis_title="convergence",
        title=f"Convergence-Rate of module L1 with exponential decay of gammas: {gammas}"
    )
    fig.show()

    # Plot descent path for gamma=0.95
    for i, L in enumerate([L1, L2]):
        callback, weights, vals = get_gd_state_recorder_callback()
        GradientDescent(ExponentialLR(eta, 0.95), callback=callback).fit(L(init), None, None)
        fig = plot_descent_path(L, np.asarray(weights), title=f"with model: L{i + 1} and gamma: 0.95")
        # fig.show()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    GD = GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4))

    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = np.asarray(X_train), np.asarray(y_train), np.asarray(X_test), np.asarray(y_test)

    # Plotting convergence rate of logistic regression over SA heart disease data
    module = LogisticRegression(solver=GD).fit(X_train, y_train)

    y_prob = module.predict_proba(X_train)

    fp, tp, thresholds = roc_curve(y_train, y_prob)

    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fp, y=tp, mode='markers+lines', text=thresholds, name="", showlegend=False,
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")])
    fig.update_layout(dict(title=rf"$\text{{ROC Curve Of Fitted Model, AUC}}={auc(fp, tp):.3f}$",
                           xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                           yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    fig.show()

    best_alpha = thresholds[np.argmax(tp-fp)]
    print(f"The best alpha for TP & FP rations is: {best_alpha}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    for L in ["l1", "l2"]:
        validation_score_err = []

        for lam in lambdas:
            module = LogisticRegression(solver=GD, penalty=L, lam=lam)
            train_err, valid_err = cross_validate(module, X_train, y_train, MCE)
            validation_score_err.append(valid_err)

        best_lambda = lambdas[np.argmin(validation_score_err)]
        best_lam_module = LogisticRegression(solver=GD, penalty=L, lam=best_lambda).fit(X_train, y_train)
        test_error = best_lam_module.loss(X_test, y_test)
        print(f"Module {L} receive the lowest test-error of: {test_error}, by the lambda: {best_lambda}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()


