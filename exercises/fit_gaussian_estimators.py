from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():

    # Question 1 - Draw samples and print fitted model
    x_vals = np.random.normal(10, 1, 1000)
    ug = UnivariateGaussian()
    ug.fit(x_vals)
    print(f"({ug.mu_}, {ug.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    mu = 1
    estimated_expectations = []
    sample_size = np.arange(10, 1010, 10)
    for size in sample_size:
        mu_after_fit = ug.fit(x_vals[:size]).mu_
        estimated_expectations.append(np.abs(mu - mu_after_fit))

    go.Figure([go.Scatter(x=sample_size, y=estimated_expectations,
                          mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(title=r"$\text{Absolute distance between "
                                     r"estimated and true value of the "
                                     r"expectation, as a function of the "
                                     r"sample size}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="$\\text{estimated "
                                           "expectations}$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model

    pdf_vals = ug.pdf(x_vals)
    go.Figure([go.Scatter(x=x_vals, y=pdf_vals, mode='markers',
                          line=dict(width=4, color="rgb(204,68,83)"),
                          name=r'$N(\mu, \frac{\sigma^2}{m1})$')],
              layout=go.Layout(barmode='overlay',
                               title=r"$\text{PDF values as a function of "
                                     r"the drawn samples under the fitted "
                                     r"model}$",
                               xaxis_title="$\\text{sample  values}$",
                               yaxis_title="PDF values")).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                        [0.2, 2, 0, 0],
                        [0, 0, 1, 0],
                        [0.5, 0, 0, 1]])
    x_vals = np.random.multivariate_normal(mu, sigma, 1000)
    mvg = MultivariateGaussian()
    mvg.fit(x_vals)

    print(mvg.mu_)
    print(mvg.cov_)

    # Question 5 - Likelihood evaluation
    space_vals = np.linspace(-10, 10, 200)
    likelihood_vals = np.zeros((space_vals.size, space_vals.size))
    for i in range(space_vals.size):
        for j in range(space_vals.size):
            m = np.array([space_vals[i], 0, space_vals[j], 0])
            likelihood_vals[i, j] = mvg.log_likelihood(m.transpose(), sigma,
                                                       x_vals)

    hm = go.Figure().add_heatmap(x=space_vals, y=space_vals,
                                     z=likelihood_vals)
    hm.update_layout(title="Heatmap of calculated log-likelihood values for "
                           "models with expectation based on f1 and f3 "
                           "values",
                     xaxis_title="f3 values", yaxis_title="f1 values")
    hm.show()


    # Question 6 - Maximum likelihood
    f1, f3 = np.unravel_index(np.argmax(likelihood_vals, axis=None),
                              likelihood_vals.shape)
    print(f"(max likelihood: {likelihood_vals[f1, f3]}, f1 value for max: "
          f"{space_vals[f1]}, f3 value for max: {space_vals[f3]})")

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
