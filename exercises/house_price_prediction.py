from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    data = pd.read_csv(filename).dropna()
    data.drop(data[(data["id"] == 0)].index, inplace=True)
    features = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot",
                "waterfront", "view", "condition", "grade", "sqft_above",
                "sqft_basement"]
    floors = pd.get_dummies(data["floors"])
    house_age = data["yr_built"] - pd.to_numeric(data["date"].astype(
        str).apply(lambda x: x[:4]))
    years_from_renovation = data["yr_renovated"] - pd.to_numeric(data[
                                 "date"].astype(str).apply(lambda x: x[:4]))
    last_renovation_or_built_year = pd.concat([house_age,
                                               years_from_renovation],
                                              axis=1).max(axis=1)
    data["zipcode"] = (data["zipcode"] / 10).astype(int)
    zipcodes = pd.get_dummies(data["zipcode"], prefix="zipcode-")
    x = pd.concat([floors, data[features], zipcodes], axis=1)
    x.insert(0, "house_age", house_age)
    x.insert(0, "last_renovation_or_built_year", last_renovation_or_built_year)
    return (x, data["price"])


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str =
".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for col in X.columns:
        feature = X[col]
        pearson_correlation = np.cov(feature, y)[0, 1] / (np.sqrt(np.var(
            feature) * np.var(y)))
        layout = go.Layout(title=f"Pearson Correlation between {col} and "
                                 f"response: {pearson_correlation}",
                           xaxis={"title": f"x - {col} values"},
                           yaxis={"title": "y - response values"})
        fig = go.Figure([go.Scatter(x=feature, y=y, mode="markers")],
                        layout=layout)
        fig.show()
        fig.write_image(f"{output_path}{col}.png", format="png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    x, y = load_data("C:\\Users\\rotem\\IML.HUJI\\datasets\\house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(x, y, "../")

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(x, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10
    # times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of
    # size (mean-2*std, mean+2*std)
    precentage_range = np.arange(10, 101, 1)
    std_vals = []
    mean_vals = []
    linearregression = LinearRegression()

    for p in precentage_range:
        losses = []
        for i in range(10):
            p_train_x, p_train_y, p_test_x, p_test_y = split_train_test(x, y,
                                                                (p / 100))
            linearregression.fit(p_train_x.to_numpy(), p_train_y.to_numpy())
            losses.append(
            linearregression.loss(test_x.to_numpy(), test_y.to_numpy()))
        mean_vals.append(np.mean(losses))
        std_vals.append(np.std(losses))

    mean_vals = np.array(mean_vals)
    std_vals = np.array(std_vals)

    fig = go.Figure(
        [go.Scatter(x=precentage_range, y=mean_vals, mode="markers+lines",
                    name="Mean Prediction", line=dict(dash="dash"),
                    marker=dict(color="red", opacity=.7)),
         go.Scatter(x=precentage_range, y=mean_vals - 2 * std_vals,
                    fill=None, mode="lines",
                    line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=precentage_range, y=mean_vals + 2 * std_vals,
                    fill='tonexty', mode="lines",
                    line=dict(color="lightgrey"),
                    showlegend=False), ],
        layout=go.Layout(
            title=r"$\text{The mean loss as function of p%}$",
            xaxis_title="$\\text{percentage}$",
            yaxis_title="$\\text{mean loss}$"))
    fig.show()