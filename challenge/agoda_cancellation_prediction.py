import itertools

from sklearn.model_selection import train_test_split

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator

import matplotlib.pyplot as plt
from sklearn import metrics

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


def agoda_preprocessor(full_data: np.ndarray):
    # fill cancellation datetime which doesn't exist as 0
    full_data.loc[full_data["cancellation_datetime"].isnull(), "cancellation_datetime"] = full_data["checkin_date"]
    full_data['cancellation_datetime'] = pd.to_datetime(full_data["cancellation_datetime"])

    features = data_preprocessor(full_data)
    full_data["cancel_warning_days"] = (full_data['checkin_date'] - full_data['cancellation_datetime']).dt.days
    full_data["days_cancelled_after_booking"] = (full_data["cancellation_datetime"] - full_data["booking_datetime"]).dt.days

    labels = (7 <= full_data["days_cancelled_after_booking"]) & (full_data["days_cancelled_after_booking"] <= 43)
    return features, np.asarray(labels).astype(int)


def load_agoda_dataset():
    """
    Load Agoda booking cancellation dataset

    Returns
    -------
    Design matrix and response vector in the following format:
    - Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """

    # clean data for unrealistic shit
    full_data = pd.read_csv("../datasets/agoda_cancellation_train.csv").drop_duplicates()

    features, labels = agoda_preprocessor(full_data)

    return features, labels


def data_preprocessor(full_data):
    # starting with the numerical and boolean columns
    features = full_data[["hotel_star_rating",
                          "guest_is_not_the_customer",
                          "original_selling_amount",
                          "is_user_logged_in",
                          "is_first_booking",
                          "cancellation_policy_code",
                          ]].fillna(0)

    # how much the customer cares about his order, sums all it's requests
    features["num_requests"] = (full_data["request_nonesmoke"].fillna(0) +
                                full_data["request_latecheckin"].fillna(0) +
                                full_data["request_highfloor"].fillna(0) +
                                full_data["request_largebed"].fillna(0) +
                                full_data["request_twinbeds"].fillna(0) +
                                full_data["request_airport"].fillna(0) +
                                full_data["request_earlycheckin"].fillna(0))

    features["charge_option"] = full_data["charge_option"].apply(lambda x: 1 if x == "Pay Later" else 0)

    # accom = {"":}
    # features["accommadation_type_name"] = full_data["accommadation_type_name"].apply(lambda x: accom[x])

    full_data['booking_datetime'] = pd.to_datetime(full_data['booking_datetime'])
    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date'])
    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date'])

    # add date connected numerical columns
    features["days_to_checkin"] = (full_data["checkin_date"] - full_data["booking_datetime"]).dt.days
    features["num_nights"] = (full_data['checkout_date'] - full_data['checkin_date']).dt.days - 1

    # deal with cancellation policy code
    features['parsed_cancellation'] = features.apply(lambda x: cancel_parser(x['cancellation_policy_code'], x['num_nights']), axis=1)
    features[['cd1', 'cp1', 'cd2', 'cp2', 'ns']] = pd.DataFrame(features['parsed_cancellation'].tolist(), index=features.index)
    del features["cancellation_policy_code"]
    del features['parsed_cancellation']

    return features


def cross_validate(estimator, X: np.ndarray, y: np.ndarray, cv):
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    cv: int
        Specify the number of folds.

    Returns
    -------

    validation_score: float
        Average validation score over folds
    """
    validation_scores = []
    split_X, split_y = np.array_split(X, cv), np.array_split(y, cv)

    for i in range(cv):
        # create S\Si & Si
        train_x, train_y = np.concatenate(np.delete(split_X, i, axis=0)), np.concatenate(np.delete(split_y, i, axis=0))
        test_x, test_y = split_X[i], split_y[i]

        # fit the estimator to the current folds
        A = estimator.fit(train_x, train_y)

        # predict over the validation fold and over the hole train set
        validation_scores.append(metrics.f1_score(A.predict(test_x), test_y, average='macro'))

    return np.array(validation_scores).mean()


def training_playground(X, y):
    """
    Evaluate current model performances over previous weeks datasets.

    Parameters
    ----------
    X: the previous weeks unite dataset
    y: the previous weeks unite labels

    """

    # f1_scores = []
    # for true, false in itertools.product(list(np.arange(0.6, 1, 0.05)), list(np.arange(0.03, 0.1, 0.01))):
    #     print(true, false)
    #     estimator = AgodaCancellationEstimator(true, false)
    #     f1_scores.append(cross_validate(estimator, X, y, cv=6))
    #
    # print(f1_scores)

    # define train & test sets.
    train_X, test_X, train_y, test_y = train_test_split(X.to_numpy(), y.to_numpy(), test_size=1/6)

    # Fit model over data
    prev_estimator = AgodaCancellationEstimator(0.6, 0.07).fit(train_X, train_y)

    # Predict for test_X
    y_pred = pd.DataFrame(prev_estimator.predict(test_X), columns=["predicted_values"])

    # confusion matrix
    cm = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(test_y, y_pred))
    cm.plot()
    plt.show()

    # Performances:
    print("Area Under Curve: ", metrics.roc_auc_score(test_y, y_pred))
    print("Accuracy: ", metrics.accuracy_score(test_y, y_pred))
    print("Recall: ", metrics.recall_score(test_y, y_pred))
    print("Precision: ", metrics.precision_score(test_y, y_pred))
    print("F1 Macro Score: ", metrics.f1_score(test_y, y_pred, average='macro'))


def evaluate_and_export(X, y, test_csv_filename):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    X: the previous weeks unite dataset
    y: the previous weeks unite labels

    test_csv_filename: path to the current week test-set csv file

    """
    f1_scores = []
    range_of_weights = list(itertools.product(list(np.arange(0.6, 1, 0.05)), list(np.arange(0.03, 0.1, 0.01))))
    for true, false in range_of_weights:
        estimator = AgodaCancellationEstimator(true, false)
        f1_scores.append(cross_validate(estimator, X, y, cv=6))

    print(np.max(f1_scores))

    true_weight, false_weight = range_of_weights[np.argmax(f1_scores)]

    # Fit model over data
    prev_estimator = AgodaCancellationEstimator(true_weight, false_weight).fit(X, y)

    # Store model predictions over test set
    test_set = pd.read_csv(test_csv_filename).drop_duplicates()

    # predict over current-week test-set
    X = data_preprocessor(test_set)
    y_pred = pd.DataFrame(prev_estimator.predict(X), columns=["predicted_values"])

    # export the current-week predicted labels
    pd.DataFrame(y_pred, columns=["predicted_values"]).to_csv("342473642_206200552_316457340.csv", index=False)


def load_previous():
    """
    Load Previous-weeks test-sets and labels

    Returns
    -------
    Design matrix and response vector in the following format:
    - Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    data_set = pd.read_csv(f'testsets//t1.csv')
    data_set['label'] = pd.read_csv(f'labels//l1.csv')["cancel"]
    for i in range(2, 7):
        ti = pd.read_csv(f'testsets//t{i}.csv')
        li = pd.read_csv(f'labels//l{i}.csv')["cancel"]
        ti['label'] = li
        data_set = pd.concat([data_set, ti])
    full_data = data_set.drop_duplicates()

    labels = full_data['label'].astype(int)
    features = data_preprocessor(full_data.drop('label', axis=1))

    # canceled = full_data.drop('label', axis=1)[labels == 1]

    return features, labels


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, labels = load_previous()

    # training_playground(df, labels)

    evaluate_and_export(df, labels, "testsets/t7.csv")