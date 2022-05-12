import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size
    Parameters
    ----------
    n: int
        Number of samples to generate
    noise_ratio: float
        Ratio of labels to invert
    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples
    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape 
    (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), \
                                           generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    learners_range = [i for i in range(1, n_learners + 1)]
    adaboost_model = AdaBoost(wl=DecisionStump, iterations=n_learners)
    adaboost_model.fit(train_X, train_y)
    train_losses = []
    test_losses = []
    for i in range(1, n_learners):
        train_losses.append(adaboost_model.partial_loss(train_X, train_y, i))
        test_losses.append(adaboost_model.partial_loss(test_X, test_y, i))
    fig = go.Figure()
    fig.add_traces([go.Scatter(x=learners_range, y=train_losses,
                               name="train"),
                    go.Scatter(x=learners_range, y=test_losses,
                               name="test")])
    fig.update_layout(title="Training and test error as a function of the "
                            f"number of fitted learners, noise = {noise}",
                      xaxis_title="number of fitted learners",
                      yaxis_title="error values")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X,
                               test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig2 = make_subplots(rows=2, cols=2, subplot_titles=[
                            rf"$\textbf{{ensemble up to iteration {j}}}$"
                            for j in T], horizontal_spacing=0.05,
                        vertical_spacing=0.1)
    for i, t in enumerate(T):
        prediction_func = lambda x: adaboost_model.partial_predict(x, t)
        fig2.add_traces([decision_surface(prediction_func, lims[0], lims[1],
                         showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                   mode="markers", showlegend=False,
                                   marker=dict(color=test_y,
                                               colorscale=[custom[0],
                                                           custom[-1]],
                                   line=dict(color="black", width=1)))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig2.update_layout(title=rf"$\textbf{{Decision Boundaries by using "
                    "the ensemble up to different iterations, noise ="
                             f" {noise}}}$")
    fig2.show()

    # Question 3: Decision surface of best performing ensemble
    min_loss_ind = np.argmin(test_losses, axis=0) + 1
    predictions_min_loss = adaboost_model.partial_predict(test_X, min_loss_ind)
    min_loss_accuracy = accuracy(test_y, predictions_min_loss)
    predictions_according_to_min_loss = lambda x: \
        adaboost_model.partial_predict(x, min_loss_ind)
    fig3 = go.Figure(
        [decision_surface(predictions_according_to_min_loss,
                          lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                    mode="markers", showlegend=False,
                    marker=dict(color=test_y, colorscale=[custom[0],
                                                          custom[-1]],
                                line=dict(color="black", width=1)))])

    fig3.update_layout(title=rf"$\textbf{{Decision surface of the ensemble "
    f"of size {min_loss_ind} that achieved the lowest test error, and it's " \
    f"accuracy is {min_loss_accuracy}, noise = {noise}}}$")

    fig3.show()

    # Question 4: Decision surface with weighted samples
    D = adaboost_model.D_ / np.max(adaboost_model.D_) * 5
    fig4 = go.Figure(
        [decision_surface(adaboost_model.predict, lims[0], lims[1],
                          showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                    mode="markers", showlegend=False,
                    marker=dict(color=train_y, colorscale=[custom[0],
                                                           custom[-1]], size=D,
                                line=dict(color="black", width=1)))])

    fig4.update_layout(title=rf"$\textbf{{Training set with a point size "
    f"proportional to it’s weight and color indicating its label, noise "
                             f"= {noise}}}$")

    fig4.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
