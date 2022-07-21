import numpy as np
from typing import List, Union, NoReturn
from IMLearn.base.base_module import BaseModule
from IMLearn.base.base_estimator import BaseEstimator
from IMLearn.desent_methods import StochasticGradientDescent, GradientDescent
from .modules import FullyConnectedLayer


class NeuralNetwork(BaseEstimator, BaseModule):
    """
    Class representing a feed-forward fully-connected neural network

    Attributes:
    ----------
    modules_: List[FullyConnectedLayer]
        A list of network layers, each a fully connected layer with its specified activation function

    loss_fn_: BaseModule
        Network's loss function to optimize weights with respect to

    solver_: Union[StochasticGradientDescent, GradientDescent]
        Instance of optimization algorithm used to optimize network

    pre_activations_:
    """
    def __init__(self,
                 modules: List[FullyConnectedLayer],
                 loss_fn: BaseModule,
                 solver: Union[StochasticGradientDescent, GradientDescent]):
        super().__init__()
        self.modules_ = modules
        self.loss_fn_ = loss_fn
        self.solver_ = solver

        self.pre_activations_ = None  # values before activation (w times samples) for each module
        self.post_activations_ = None  # values after activation for each module

    # region BaseEstimator implementations
    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit network over given input data using specified architecture and solver

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.solver_.fit(f=self, X=X, y=y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels for given samples using fitted network

        Parameters:
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted labels of given samples
        """
        # choose class with highest probability
        return np.argmax(self.compute_prediction(X=X), axis=1)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculates network's loss over given data

        Parameters
        -----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        --------
        loss : float
            Performance under specified loss function
        """
        return self.loss_fn_.compute_output(X=self.compute_prediction(X=X), y=y)
    # endregion

    # region BaseModule implementations
    def compute_output(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network output with respect to modules' weights given input samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        output: ndarray of shape (1,)
            Network's output value including pass through the specified loss function

        Notes
        -----
        Function stores all intermediate values in the `self.pre_activations_` and `self.post_activations_` arrays
        """
        # compute network output
        X = self.compute_prediction(X)

        return self.loss_fn_.compute_output(X=X, y=y)

    def compute_prediction(self, X: np.ndarray):
        """
        Compute network output (forward pass) with respect to modules' weights given input samples, except pass
        through specified loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        Returns
        -------
        output : ndarray of shape (n_samples, n_classes)
            Network's output values prior to the call of the loss function
        """
        self.pre_activations_ = []
        self.post_activations_ = [X]  # for backpropagation X is considered the first layer's input

        # pass X through network
        for module in self.modules_:
            # add bias term to input
            X_ = np.insert(X, 0, 1, axis=1) if module.include_intercept_ else X
            self.pre_activations_.append(X_ @ module.weights)

            X = module.compute_output(X=X)
            self.post_activations_.append(X)

        return X

    def compute_jacobian(self, X: np.ndarray, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute network's derivative (backward pass) according to the backpropagation algorithm.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        A flattened array containing the gradients of every learned layer.

        Notes
        -----
        Function depends on values calculated in forward pass and stored in
        `self.pre_activations_` and `self.post_activations_`
        """
        # start forward pass and set up pre_activations_ and post_activations_
        self.compute_output(X=X, y=y, **kwargs)
        partials = []

        # initialize delta value (used to save up calculations in backpropagation)
        delta_T = self.loss_fn_.compute_jacobian(X=self.post_activations_[-1], y=y)
        delta_T = np.einsum('kjl,kl->kj', self.modules_[-1].activation_.compute_jacobian(X=self.pre_activations_[-1]),
                            delta_T)

        # backpropagate from last layer to first layer, note the index i starts from 1
        for i, module in enumerate(reversed(self.modules_), start=1):
            # calculate the derivative of objective with respect to the weights of the current layer
            if module.include_intercept_:  # add bias term
                final_partial = np.einsum('ki,kj->kij', delta_T, np.concatenate(
                    (np.ones((self.post_activations_[-i - 1].shape[0], 1)), self.post_activations_[-i - 1]), axis=1))
            else:
                final_partial = np.einsum('ki,kj->kij', delta_T, self.post_activations_[-i - 1])

            partials.append(final_partial)  # save the derivative

            # update delta value

            if i < len(self.modules_):  # in the last iter we don't update delta (+we can't, index error)
                activation_derivative = self.modules_[-i - 1].activation_.compute_jacobian(
                    X=self.pre_activations_[-i - 1])

                if module.include_intercept_:  # if there's a bias, don't use it since it doesn't affect previous layers
                    weights_times_delta = np.einsum('il,kl->ki', module.weights[1:, :], delta_T)
                    delta_T = np.einsum('kil,kl->ki', activation_derivative, weights_times_delta)
                else:
                    weights_times_delta = np.einsum('il,kl->ki', module.weights, delta_T)
                    delta_T = np.einsum('kil,kl->ki', activation_derivative, weights_times_delta)

        # reverse partials to get correct order
        partials = partials[::-1]

        # get the average of the derivatives and transpose for correct shape
        for i in range(len(partials)):
            partials[i] = np.mean(partials[i], axis=0).T

        # return the flattened array
        return self._flatten_parameters(partials)

    @property
    def weights(self) -> np.ndarray:
        """
        Get flattened weights vector. Solvers expect weights as a flattened vector

        Returns
        --------
        weights : ndarray of shape (n_features,)
            The network's weights as a flattened vector
        """
        return NeuralNetwork._flatten_parameters([module.weights for module in self.modules_])

    @weights.setter
    def weights(self, weights) -> None:
        """
        Updates network's weights given a *flat* vector of weights. Solvers are expected to update
        weights based on their flattened representation. Function first un-flattens weights and then
        performs weights' updates throughout the network layers

        Parameters
        -----------
        weights : np.ndarray of shape (n_features,)
            A flat vector of weights to update the model
        """
        non_flat_weights = NeuralNetwork._unflatten_parameters(weights, self.modules_)
        for module, weights in zip(self.modules_, non_flat_weights):
            module.weights = weights
    # endregion

    # region Internal methods
    @staticmethod
    def _flatten_parameters(params: List[np.ndarray]) -> np.ndarray:
        """
        Flattens list of all given weights to a single one dimensional vector. To be used when passing
        weights to the solver

        Parameters
        ----------
        params : List[np.ndarray]
            List of differently shaped weight matrices

        Returns
        -------
        weights: ndarray
            A flattened array containing all weights
        """
        return np.concatenate([grad.flatten() for grad in params])

    @staticmethod
    def _unflatten_parameters(flat_params: np.ndarray, modules: List[BaseModule]) -> List[np.ndarray]:
        """
        Performing the inverse operation of "flatten_parameters"

        Parameters
        ----------
        flat_params : ndarray of shape (n_weights,)
            A flat vector containing all weights

        modules : List[BaseModule]
            List of network layers to be used for specifying shapes of weight matrices

        Returns
        -------
        weights: List[ndarray]
            A list where each item contains the weights of the corresponding layer of the network, shaped
            as expected by layer's module
        """
        low, param_list = 0, []
        for module in modules:
            r, c = module.shape
            high = low + r * c
            param_list.append(flat_params[low: high].reshape(module.shape))
            low = high
        return param_list
    # endregion
