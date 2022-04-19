import sklearn

from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator

import numpy as np
import pandas as pd
import re

PATTERN = re.compile(r"((?P<days1>[1-9]\d*)D(?P<amount1>[1-9]\d*[NP])_)?((?P<days2>[1-9]\d*)D(?P<amount2>[1-9]\d*[NP])_)?(?P<noshow>[1-9]\d*[NP])?")


def cancel_parser(policy: str, nights_num):
    if nights_num <= 0:
        nights_num = 1
    match = PATTERN.match(policy)
    if match is None:
        return policy
    else:
        noshow = match.group("noshow")
        noshow = 1 if noshow is None else int(noshow[:-1])/100 if noshow[-1] == 'P' else int(noshow[:-1]) / nights_num

        days1 = match.group("days1")
        if days1 is None:
            days1 = 0
            amount1 = noshow
        else:
            days1 = int(days1)
            amount1 = match.group("amount1")
            amount1 = int(amount1[:-1])/100 if amount1[-1] == 'P' else int(amount1[:-1])/nights_num

        days2 = match.group("days2")
        if days2 is None:
            days2 = 0
            amount2 = amount1
        else:
            days2 = int(days2)
            amount2 = match.group("amount2")
            amount2 = int(amount2[:-1])/100 if amount2[-1] == 'P' else int(amount2[:-1])/nights_num

        return days1, amount1, days2, amount2, noshow


def training_preprocessor(full_data: np.ndarray):
    # fill cancellation datetime which doesn't exist as 0
    full_data.loc[full_data["cancellation_datetime"].isnull(), "cancellation_datetime"] = full_data["checkin_date"]
    full_data['cancellation_datetime'] = pd.to_datetime(full_data["cancellation_datetime"])

    features = testing_preprocessor(full_data)
    full_data["cancel_warning_days"] = (full_data['checkin_date'] - full_data['cancellation_datetime']).dt.days
    full_data["days_cancelled_after_booking"] = (full_data["cancellation_datetime"] - full_data["booking_datetime"]).dt.days

    # todo find relationship between cancellation policy date, P  and cancel date
    labels = (7 <= full_data["days_cancelled_after_booking"]) & (full_data["days_cancelled_after_booking"] <= 43)
    # labels = (full_data["cancel_warning_days"] != 0)
    return features, labels


def testing_preprocessor(full_data):
    # starting with the numerical and boolean columns
    features = full_data[["hotel_star_rating",
                          "guest_is_not_the_customer",
                          "original_selling_amount",
                          "is_user_logged_in",
                          "is_first_booking",
                          "cancellation_policy_code"
                          ]].fillna(0)

    # how much the customer cares about his order
    features["num_requests"] = (full_data["request_nonesmoke"].fillna(0) +
                                full_data["request_latecheckin"].fillna(0) +
                                full_data["request_highfloor"].fillna(0) +
                                full_data["request_largebed"].fillna(0) +
                                full_data["request_twinbeds"].fillna(0) +
                                full_data["request_airport"].fillna(0) +
                                full_data["request_earlycheckin"].fillna(0))

    # change to numerical
    for f in ["is_user_logged_in", "is_first_booking"]:
        features[f] = features[f].astype(int)

    full_data['booking_datetime'] = pd.to_datetime(full_data['booking_datetime'])
    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date'])
    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date'])

    # add date connected numerical columns
    features["days_to_checkin"] = (full_data["checkin_date"] - full_data["booking_datetime"]).dt.days
    features["num_nights"] = (full_data['checkout_date'] - full_data['checkin_date']).dt.days - 1

    # deal with cancellation policy code
    features['B'] = features.apply(lambda x: cancel_parser(x['cancellation_policy_code'], x['num_nights']), axis=1)
    features[['cd1', 'cp1', 'cd2', 'cp2', 'ns']] = pd.DataFrame(features['B'].tolist(), index=features.index)
    del features["cancellation_policy_code"]
    del features['B']

    return features


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """

    # clean data for unrealistic shit
    full_data = pd.read_csv(filename).drop_duplicates()

    features, labels = training_preprocessor(full_data)

    return features, labels


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str, test_y):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    test_y
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    y_pred = pd.DataFrame(estimator.predict(X), columns=["predicted_values"])
    pd.DataFrame(y_pred, columns=["predicted_values"]).to_csv(filename, index=False)
    # print("Area Under Curve: ", sklearn.metrics.roc_auc_score(test_y, y_pred))
    # print("Accuracy: ", sklearn.metrics.accuracy_score(test_y, y_pred))
    # print("Recall: ", sklearn.metrics.recall_score(test_y, y_pred))
    # print("Precision: ", sklearn.metrics.precision_score(test_y, y_pred))


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data("../datasets/agoda_cancellation_train.csv")

    # train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(df, cancellation_labels, test_size=0.2)

    # Fit model over data
    estimator = AgodaCancellationEstimator().fit(df, cancellation_labels)

    # Store model predictions over test set
    test_set = pd.read_csv("test_set_week_2.csv").drop_duplicates()
    evaluate_and_export(estimator, testing_preprocessor(test_set), "342473642_206200552_316457340.csv", 0)
