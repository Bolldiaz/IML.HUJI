from __future__ import annotations
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator:
       Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_scores, validation_scores = [], []
    split_X, split_y = np.array_split(X, cv), np.array_split(y, cv)

    for i in range(cv):
        # create S\Si & Si
        train_x, train_y = np.concatenate(np.delete(split_X, i, axis=0)), np.concatenate(np.delete(split_y, i, axis=0))
        test_x, test_y = split_X[i], split_y[i]

        # fit the estimator to the current folds
        A = estimator.fit(train_x, train_y)

        # predict over the validation fold and over the hole train set
        validation_scores.append(scoring(A.predict(test_x), test_y))
        train_scores.append(scoring(A.predict(X), y))

    return np.array(train_scores).mean(), np.array(validation_scores).mean()
