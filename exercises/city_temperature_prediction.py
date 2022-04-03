import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"]).dropna().drop_duplicates()
    df['DayOfYear'] = df['Date'].dt.dayofyear

    df = df[df["Month"].isin(range(1, 13)) &
            df["Day"].isin(range(1, 32))]

    df = df[df["Year"] > 0]
    df = df[df["Temp"] > -20]

    return df


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data("ex2/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    df["Year"] = df["Year"].astype(str)
    israel_df = df[df["Country"] == 'Israel']
    graph = px.scatter(israel_df, x="DayOfYear", y="Temp", color="Year",
                       title="Israel average daily temperature",
                       labels={"DayOfYear": 'Day Of the Year', "Temp": "Temp"})
    graph.write_image("ex2/Israel_temp.png")

    monthly_std = israel_df.groupby('Month').agg({'Temp': 'std'})
    graph = px.bar(monthly_std, title="Israel daily temperature std per month.", labels={"Temp": "temp std"})
    graph.write_image("ex2/monthly_std.png")
    df["Year"] = df["Year"].astype(int)

    # Question 3 - Exploring differences between countries
    country_monthly_mean_std = df.groupby(by=["Month", "Country"]).Temp.agg(['mean', 'std']).reset_index()
    graph = px.line(country_monthly_mean_std, x='Month', y='mean', error_y='std', color='Country',
                    labels={'y': "mean with error of std"}, title="Average monthly temperature, with error bars of std")
    graph.write_image("ex2/country_monthly_mean_std.png")

    # Question 4 - Fitting model for different values of `k`
    trainX, trainY, testX, testY = split_train_test(israel_df['DayOfYear'], israel_df['Temp'])
    loss_records = []
    for k in range(1, 11):
        polynomial_fitting = PolynomialFitting(k)
        polynomial_fitting.fit(trainX, trainY)
        loss_val = polynomial_fitting.loss(testX, testY)
        loss_records.append(loss_val)
        print(f"Test loss of k={k}: {format(loss_val, '.2f')}")

    loss_records = np.around(loss_records, 2)
    graph = px.bar(x=range(1, 11), y=loss_records,
                   title="Temp fitting loss as function of k.", labels={"x": "polynom degree", "y":"loss"})
    graph.write_image("ex2/loss_of_k.png")

    # Question 5 - Evaluating fitted model on different countries
    polynomial_fitting = PolynomialFitting(5)
    polynomial_fitting.fit(israel_df['DayOfYear'], israel_df['Temp'])

    countries = list(set(df["Country"].values))
    loss_records = []
    for country in countries:
        country_df = df[df["Country"] == country]
        loss_records.append(polynomial_fitting.loss(country_df['DayOfYear'], country_df['Temp']))

    loss_records = np.around(loss_records, 2)
    graph = px.bar(x=countries, y=loss_records, labels={"x": "country", "y": "loss"},
                   title="Loss after fitting of each country.")
    graph.write_image("ex2/loss_of_countries.png")






