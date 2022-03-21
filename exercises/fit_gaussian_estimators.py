from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.express as pex
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    un = UnivariateGaussian()
    X = np.random.normal(10.0, 1.0, 1000)
    un.fit(X)
    print(f"Expectation: {un.mu_}, Variance: {un.var_}\n")

    # Question 2 - Empirically showing sample mean is consistent
    loss = []
    un = UnivariateGaussian()
    for sample_size in range(10, 1010, 10):
        un.fit(np.random.choice(X, size=sample_size, replace=False))
        loss.append(np.abs(un.mu_ - 10))

    graph = pex.line(x=range(10, 1010, 10),
                     y=np.asarray(loss),
                     labels=dict(x="Sample size", y="Loss of the expectation"),
                     title="Absolute distance between the estimated & true-value of the expectation, "
                           "as a function of the sample size")
    graph.write_html('first_figure.html', auto_open=True)

    # Question 3 - Plotting Empirical PDF of fitted model
    plt.scatter(X, un.pdf(X))
    plt.ylabel("PDF")
    plt.xlabel("SAMPLES")
    plt.title("Empirical PDF of samples")
    plt.show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    multivariate_gaussian = MultivariateGaussian()
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])

    X = np.random.multivariate_normal(mu, cov, 1000)
    multivariate_gaussian.fit(X)
    print(f"Estimated expectation:\n{multivariate_gaussian.mu_}\n\n"
          f"Estimated covariance:\n{multivariate_gaussian.cov_}\n")

    # Question 5 - Likelihood evaluation
    linspace = np.linspace(-10, 10, 200)
    log_likelihood = [[(multivariate_gaussian.log_likelihood(np.asarray([f1, 0, f3, 0]).T, cov, X))
                       for f3 in linspace]
                       for f1 in linspace]

    graph = pex.imshow(log_likelihood,
                       x=linspace,
                       y=linspace,
                       labels=dict(x="f1",
                                   y="f3",
                                   color="log-likelihood"),
                       title="log-likelihood for models with expectation "
                             "μ=[f1, 0, f3, 0] and the given true covariance matrix Σ")
    graph.write_html('first_figure.html', auto_open=True)

    # Question 6 - Maximum likelihood
    max_coord = np.where(log_likelihood == np.amax(log_likelihood))
    f1_argmax = linspace[max_coord[0][0]]
    f3_argmax = linspace[max_coord[1][0]]
    form = "{:.3f}"
    print(f"max(log-likelihood): {form.format(np.amax(log_likelihood))}")
    print(f"argmax(log-likelihood): ({form.format(f1_argmax)} : {form.format(f3_argmax)})")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
