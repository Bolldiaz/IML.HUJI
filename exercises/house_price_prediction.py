import copy

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

TEST_PERCENTAGE = 0.75

def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()


    # Removal of outliers:
    df = df[df["bedrooms"] < 20]
    df = df[df["sqft_lot"] < 1250000]
    df = df[df["sqft_lot15"] < 500000]

    # Clean samples don't hold those constraints:
    for feature in ["price", "sqft_living", "sqft_lot", "sqft_above", "yr_built", "sqft_living15", "sqft_lot15"]:
        df = df[df[feature] > 0]
    for feature in ["bathrooms", "floors", "sqft_basement", "yr_renovated"]:
        df = df[df[feature] >= 0]
    df = df[df["waterfront"].isin([0, 1]) &
            df["view"].isin(range(5)) &
            df["condition"].isin(range(1, 6)) &
            df["grade"].isin(range(1, 15))]

    # Change those values to be categorical:
    df["decade_built"] = (df["yr_built"] / 10).astype(int)
    df["zipcode"] = df["zipcode"].astype(int)
    df = pd.get_dummies(df, prefix='zipcode_', columns=['zipcode'])
    df = pd.get_dummies(df, prefix='decade_built', columns=['decade_built'])

    df = df.drop(["id", "lat", "long", "date", "yr_built", "yr_renovated"], axis=1)

    return df.drop("price", axis=1), df.price


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # remove all categorical columns (and intercept) which cause a redundant cov=0:
    X = X.loc[:, ~(X.columns.str.contains('^zipcode_') | X.columns.str.contains('^decade_built_'))].drop("intercept", axis=1)

    for feature in X:
        p = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))

        graph = px.scatter(pd.DataFrame({'x': X[feature], 'y': y}), x="x", y="y", trendline="ols",
                         title=f"Correlation Between {feature} Values and Response <br>Pearson Correlation {p}",
                         labels={"x": f"{feature} Values", "y": "Response Values"})
        graph.write_image("ex2/pearson.correlation.%s.png" % feature)


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("ex2/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    trainX, trainY, testX, testY = split_train_test(X, y, TEST_PERCENTAGE)
    train_df = trainX.merge(trainY, left_index=True, right_index=True)

    # Question 4 - Fit model over increasing percentages of the overall training data
    linear_regression = LinearRegression()
    loss_mean, loss_std = [], []

    for p in range(10, 101):
        n = round(len(trainY) * (p / 100))

        loss_samples = []
        for _ in range(10):
            sampled = train_df.sample(n)
            linear_regression.fit(sampled.drop("price", axis=1), sampled["price"])
            loss_samples.append(linear_regression.loss(testX, testY))
        loss_samples = np.asarray(loss_samples)

        loss_mean.append(loss_samples.mean())
        loss_std.append(loss_samples.std())

    loss_mean = np.asarray(loss_mean)
    loss_std = np.asarray(loss_std)
    x_range = np.linspace(10, 101)

    fig = go.Figure([go.Scatter(x=x_range,
                                y=loss_mean,
                                mode='lines'),
                     go.Scatter(x=x_range,
                                y=loss_mean + 2 * loss_std,
                                mode='lines',marker=dict(color='#444'), showlegend=False),
                     go.Scatter(x=x_range,
                                y=loss_mean - 2 * loss_std,
                                mode='lines', marker=dict(color='#444'), showlegend=False,
                                fillcolor='rgba(68,68,68,0.3)', fill='tonexty')])

    fig.update_xaxes(ticksuffix="%", title_text="percents of training-set")
    fig.update_yaxes(title_text="loss over test-set")
    fig.update_layout(title_text="average loss as function of training size with error ribbon")
    fig.write_image('ex2/loss.png')