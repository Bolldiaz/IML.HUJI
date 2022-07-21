from __future__ import annotations
from typing import Callable, Tuple
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .gradient_descent import default_callback
from .learning_rate import FixedLR


class StochasticGradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    callback_: Callable[[...], None], default=default_callback
        A callable function to be called after each update of the model while fitting to given data.
        Callable function receives as input any argument relevant for the current GD iteration. Arguments
        are specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 batch_size: int = 1,
                 callback: Callable[[...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        batch_size: int, default=1
            Number of samples to randomly select at each iteration of the SGD algorithm

        callback: Callable[[...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data.
            Callable function receives as input any argument relevant for the current GD iteration. Arguments
            are specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.batch_size_ = batch_size
        self.callback_ = callback

        self.t_ = 0

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using SGD iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
        Objective function (module) to be minimized by SGD


        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over

        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X, y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)
            - batch_indices: np.ndarray of shape (n_batch,)
                Sample indices used in current SGD iteration
        """
        self.t_ = 0  # set iteration counter to 0
        n_samples = X.shape[0]

        prev_weights = None

        while self.t_ < self.max_iter_ and (self.t_ == 0 or np.linalg.norm(f.weights - prev_weights) > self.tol_):
            prev_weights = f.weights  # save current weights
            batch_indices = np.random.choice(n_samples, self.batch_size_, replace=False)  # select random samples
            val, jac, eta = self._partial_fit(f=f, X=X[batch_indices], y=y[batch_indices],
                                              t=self.t_)  # calculate partial derivative

            # call callback function and increment iteration counter
            self.callback_(solver=self, weights=f.weights, val=val, grad=jac, t=self.t_, eta=eta,
                           delta=np.linalg.norm(f.weights - prev_weights), batch_indices=batch_indices)
            self.t_ += 1

        return f.weights

    def _partial_fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray, t: int) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform a SGD iteration over given samples

        Parameters
        ----------
        f : BaseModule
        Objective function (module) to be minimized by SGD

        X : ndarray of shape (n_batch, n_features)
            Input data to optimize module over

        y : ndarray of shape (n_batch, )
            Responses of input data to optimize module over

        t: int
            Current SGD iteration

        Returns
        -------
        val: ndarray of shape (n_features,)
            Value of objective optimized, at current position, based on given samples

        jac: ndarray of shape (n_features, )
            Jacobian on objective optimized, at current position, based on given samples

        eta: float
            learning rate used at current iteration
        """
        # calculate value of objective and jacobian according to batch
        val = f.compute_output(X=X, y=y)
        jac = f.compute_jacobian(X=X, y=y)

        eta = self.learning_rate_.lr_step(t=t)  # get learning rate
        f.weights -= eta * jac  # update weights

        return val, jac, eta
