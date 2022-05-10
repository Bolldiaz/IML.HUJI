
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

    labels = (7 <= full_data["days_cancelled_after_booking"]) & (full_data["days_cancelled_after_booking"] <= 43)
    return features, labels


def testing_preprocessor(full_data):
    # starting with the numerical and boolean columns
    features = full_data[["hotel_star_rating",
                          "guest_is_not_the_customer",
                          "original_selling_amount",
                          "is_user_logged_in",
                          "is_first_booking",
                          "cancellation_policy_code",
                          ]].fillna(0)
    #
    # # how much the customer cares about his order
    features["num_requests"] = (full_data["request_nonesmoke"].fillna(0) +
                                full_data["request_latecheckin"].fillna(0) +
                                full_data["request_highfloor"].fillna(0) +
                                full_data["request_largebed"].fillna(0) +
                                full_data["request_twinbeds"].fillna(0) +
                                full_data["request_airport"].fillna(0) +
                                full_data["request_earlycheckin"].fillna(0))

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


def training_playground(train_X1, train_y1, train_X2, train_y2):
    """
    Evaluate current model performances over previous weeks datasets.

    Parameters
    ----------
    train_X1: the previous weeks unite dataset
    train_y1: the previous weeks unite labels

    train_X2: Agoda booking cancellation dataset
    train_y2: Agoda booking cancellation labels
    """
    # define train & test sets.
    test_X = testing_preprocessor(pd.read_csv(f'testsets//t4.csv'))
    test_y = pd.DataFrame(pd.read_csv(f'labels//l4.csv').apply(lambda x: pd.Series(x.to_string()[-1], name='label'), axis=1)).astype(int)

    # Fit model over data
    prev_estimator = AgodaCancellationEstimator("complex").fit(train_X1, train_y1)
    agoda_dataset_estimator = AgodaCancellationEstimator("logistic").fit(train_X2, train_y2)

    # Predict for test_X
    y_pred1 = pd.DataFrame(prev_estimator.predict(test_X), columns=["predicted_values"])
    y_pred2 = pd.DataFrame(agoda_dataset_estimator.predict(test_X), columns=["predicted_values"])
    y_pred = np.logical_and(y_pred1.astype(int).to_numpy(), y_pred2.astype(int).to_numpy())
    print(int(np.sum(y_pred1)), int(np.sum(y_pred2)), int(np.sum(y_pred)))

    # confusion matrix
    import matplotlib.pyplot as plt
    from sklearn import metrics

    cm = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(test_y, y_pred))
    cm.plot()
    plt.show()

    # Performances:
    print("Area Under Curve: ", metrics.roc_auc_score(test_y, y_pred))
    print("Accuracy: ", metrics.accuracy_score(test_y, y_pred))
    print("Recall: ", metrics.recall_score(test_y, y_pred))
    print("Precision: ", metrics.precision_score(test_y, y_pred))
    print("F1 Macro Score: ", metrics.f1_score(test_y, y_pred, average='macro'))


def evaluate_and_export(train_X1, train_y1, train_X2, train_y2, test_csv_filename):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    train_X1: the previous weeks unite dataset
    train_y1: the previous weeks unite labels

    train_X2: Agoda booking cancellation dataset
    train_y2: Agoda booking cancellation labels

    test_csv_filename: path to the current week test-set csv file

    """
    # Fit model over data
    prev_estimator = AgodaCancellationEstimator("complex").fit(train_X1, train_y1)
    agoda_dataset_estimator = AgodaCancellationEstimator("logistic").fit(train_X2, train_y2)

    # Store model predictions over test set
    test_set = pd.read_csv(test_csv_filename).drop_duplicates()

    # predict over current-week test-set
    X = testing_preprocessor(test_set)
    y_pred1 = pd.DataFrame(prev_estimator.predict(X), columns=["predicted_values"])
    y_pred2 = pd.DataFrame(agoda_dataset_estimator.predict(X), columns=["predicted_values"])
    y_pred = np.logical_and(y_pred1.astype(int).to_numpy(), y_pred2.astype(int).to_numpy()).astype(int)
    print(int(np.sum(y_pred1)), int(np.sum(y_pred2)), int(np.sum(y_pred)))

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
    data_set['label'] = pd.DataFrame(pd.read_csv(f'labels//l1.csv').apply(lambda x: pd.Series(x.to_string()[-1], name='label'), axis=1))
    for i in range(2, 5):
        ti = pd.read_csv(f'testsets//t{i}.csv')
        li = pd.DataFrame(pd.read_csv(f'labels//l{i}.csv').apply(lambda x: pd.Series(x.to_string()[-1], name='label'), axis=1))
        ti['label'] = li
        data_set = pd.concat([data_set, ti])
    full_data = data_set.drop_duplicates()

    labels = full_data['label'].astype(int)
    features = testing_preprocessor(full_data.drop('label', axis=1))

    return features, labels


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

    features, labels = training_preprocessor(full_data)

    return features, labels


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df1, labels1 = load_previous()
    df2, labels2 = load_agoda_dataset()

    # training_playground(df1, labels1, df2, labels2)

    # evaluate_and_export(df1, labels1, df2, labels2, "testsets/t5.csv")