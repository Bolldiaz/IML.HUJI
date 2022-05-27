from __future__ import annotations
from sklearn import datasets
from IMLearn.metrics import mean_square_error as MSE
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go

fix2point = "{:.2f}"
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    x = np.linspace(-1.2, 2, n_samples)
    f = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    epsilon = np.random.normal(0, noise, n_samples)
    y = f + epsilon

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(x), pd.Series(y), 2/3)
    train_X, train_y, test_X, test_y = train_X.to_numpy()[:, 0], train_y.to_numpy(), test_X.to_numpy()[:, 0], test_y.to_numpy()

    fig = go.Figure(
        [go.Scatter(x=x, y=f, mode='lines+markers',  marker=dict(color="black"), name=r'Noiseless Values'),
         go.Scatter(x=train_X, y=train_y, mode='markers', name=r'Noisy Train values'),
         go.Scatter(x=test_X, y=test_y, mode='markers', name=r'Noisy Test values')]
    )
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    fig.update_layout(title_text=rf"$\textbf{{Model and dataset values, where samples={n_samples} and noise={noise}}}$")
    fig.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_validation_errors = [*zip(*[cross_validate(PolynomialFitting(k), train_X, train_y, MSE)
                                      for k in range(11)])]
    fig = go.Figure(
        [go.Scatter(x=np.arange(0, 11), y=train_validation_errors[0], mode='lines+markers', name=r'Training-Error'),
         go.Scatter(x=np.arange(0, 11), y=train_validation_errors[1], mode='lines+markers', name=r'Validation-Error')]
                    )
    fig.update_xaxes(title_text="polynom degree")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text=rf"$\textbf{{Average training/validation error over degree of polynom, where samples={n_samples} and noise={noise}}}$")
    fig.show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_deg = np.argmin(np.array(train_validation_errors[1]))
    error = PolynomialFitting(best_deg).fit(train_X, train_y).loss(test_X, test_y)
    print(f"With samples={n_samples} and noise={noise} the polynom of degree={best_deg} obtain the lowest error={fix2point.format(error)}, over the test set.")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    d = datasets.load_diabetes()
    X, y = pd.DataFrame(d.data), pd.Series(d.target)
    train_X, train_y, test_X, test_y = split_train_test(X, y, n_samples / d.target.size)
    train_X, train_y, test_X, test_y = train_X.to_numpy(), train_y.to_numpy(), test_X.to_numpy(), test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.001, 2, num=n_evaluations)

    # cross validation over ridge
    ridge_errors = [*zip(*[cross_validate(RidgeRegression(lam), train_X, train_y, MSE) for lam in lambdas])]

    fig = go.Figure([go.Scatter(x=lambdas, y=ridge_errors[0], mode='lines+markers', name=r'Train Error Average'),
                     go.Scatter(x=lambdas, y=ridge_errors[1], mode='lines+markers', name=r'Validation Error Average')])
    fig.update_xaxes(title_text="regularization parameter")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text=rf"$\textbf{{Ridge regularization}}$")
    fig.show()

    # cross validation over lasso
    lasso_errors = [*zip(*[cross_validate(Lasso(alpha=lam, max_iter=10000), train_X, train_y, MSE) for lam in lambdas])]

    fig = go.Figure([go.Scatter(x=lambdas, y=lasso_errors[0], mode='lines+markers', name=r'Train Error Average'),
                     go.Scatter(x=lambdas, y=lasso_errors[1], mode='lines+markers', name=r'Validation Error Average')])
    fig.update_xaxes(title_text="regularization parameter")
    fig.update_yaxes(title_text="error values")
    fig.update_layout(title_text=rf"$\textbf{{Lasso regularization}}$")
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_best_lambda = lambdas[np.argmin(ridge_errors[1])]
    lasso_best_lambda = lambdas[np.argmin(lasso_errors[1])]

    ridge = RidgeRegression(ridge_best_lambda).fit(train_X, train_y)
    ridge_err = fix2point.format(ridge.loss(test_X, test_y))

    lasso = Lasso(alpha=lasso_best_lambda, max_iter=50000).fit(train_X, train_y)
    lasso_err = fix2point.format(MSE(test_y, lasso.predict(test_X)))

    lin_reg = LinearRegression().fit(train_X, train_y)
    lin_reg_err = fix2point.format(lin_reg.loss(test_X, test_y))

    print(f"The Ridge best lambda is: {ridge_best_lambda}, which obtain over the test set an error of {ridge_err}")
    print(f"The Lasso best lambda is: {lasso_best_lambda}, which obtain over the test set an error of {lasso_err}")
    print(f"The Linear-Regression test set error is: {lin_reg_err}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
