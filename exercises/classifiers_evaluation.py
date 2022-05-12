from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
pio.templates.default = "simple_white"
from math import atan2, pi



def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers.
    File is assumed to be an ndarray of shape (n_samples, 3) where the first 2
    columns represent features and the third column the class Parameters
    ----------
    filename: str
        Path to .npy data file
    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used
    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class
    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)

def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss
    values (y-axis) as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        x, y = load_dataset(
            "C:\\Users\\rotem\\IML.HUJI\\datasets\\" + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        perc = Perceptron(callback=(lambda p, x1, y1: losses.append(p.loss(
            x, y))))
        perc.fit(x, y)

        # Plot figure
        fig = go.Figure([go.Scatter(x=list(range(1, len(losses) + 1)),
                                    y=losses, mode="lines")])
        fig.update_layout(title="Perceptron algorithm's training loss "
                             "values as a function of the training "
                                     "iterations",
                        xaxis_title="iterations",
                        yaxis_title="loss values")
        fig.show()

def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix
    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse
    cov: ndarray of shape (2,2)
        Covariance of Gaussian
    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and
    gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        x, y = load_dataset(
            "C:\\Users\\rotem\\IML.HUJI\\datasets\\" + f)

        # Fit models and predict over training set
        lda_model = LDA()
        lda_model.fit(x, y)
        predicted_lda = lda_model.predict(x)

        gnb_model = GaussianNaiveBayes()
        gnb_model.fit(x, y)
        predicted_gnb = gnb_model.predict(x)


        # Plot a figure with two subplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions on the right. Plot
        # title should specify dataset used and subplot titles should specify
        # algorithm and accuracy

        # Create subplots
        from IMLearn.metrics import accuracy
        lda_accuracy = accuracy(y, predicted_lda)
        gnb_accuracy = accuracy(y, predicted_gnb)

        models = [("Gaussian Naive Bayes", gnb_accuracy), ("LDA",
                                                           lda_accuracy)]

        fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Model: {model}, "
                           f"Accuracy: {acc}" for model, acc in models],
                            horizontal_spacing=0.01, vertical_spacing=.03)

        # Add traces for data-points setting symbols and colors
        fig.add_trace(go.Scatter(x=x[:,0], y=x[:,1], mode="markers",
                                 marker=go.scatter.Marker(
                                 color=predicted_gnb, symbol=np.array(y))),
                      row=1,
                      col=1)
        fig.add_trace(go.Scatter(x=x[:,0], y=x[:,1], mode="markers",
                                 marker=go.scatter.Marker(
                                 color=predicted_lda, symbol=np.array(y))),
                      row=1,
                      col=2)

        fig.update_layout(title=rf"$\textbf{{{f[:9]} dataset}}$", margin=dict(
            t=100)).update_xaxes(visible=False).update_yaxes(visible=False)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(go.Scatter(x=gnb_model.mu_[:,0], y=gnb_model.mu_[:,1],
                                 mode="markers", marker=dict(color="Black",
                                             symbol="x-dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=lda_model.mu_[:,0], y=lda_model.mu_[
                                                         :,1],
                                 mode="markers", marker=dict(color="Black",
                                             symbol="x-dot")), row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(gnb_model.classes_)):
            fig.add_trace(get_ellipse(gnb_model.mu_[i], np.diag(
                gnb_model.vars_[i])), row=1, col=1)
        for i in range(len(lda_model.classes_)):
            fig.add_trace(get_ellipse(lda_model.mu_[i], lda_model.cov_),
                          row=1, col=2)

        fig.show()

if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()

