import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.

    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    to_datetime = lambda x: pd.to_datetime(x)
    data = pd.read_csv(filename, parse_dates=['Date'],
                       date_parser=to_datetime).dropna().drop_duplicates()
    data = data[data["Temp"] > -72]
    day_of_year = data['Date'].apply(lambda x: x.timetuple().tm_yday)
    data.insert(3, "DayOfYear", day_of_year)
    response = data["Temp"]
    data = data.drop("Temp", axis=1)
    return data, response

if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    design_matrix, response = load_data("C:\\Users\\rotem\\IML.HUJI\\datasets"
                        "\\City_Temperature.csv")
    data = pd.concat([design_matrix, response], axis=1)

    # Question 2 - Exploring data for specific country
    israel_data = design_matrix[design_matrix['Country'] == 'Israel']
    israel_temp = response.reindex_like(israel_data)
    string_year = israel_data["Year"].apply(lambda x: str(x))
    israel_data.insert(0, "StringYear", string_year)
    israel_fig = px.scatter(israel_data, x="DayOfYear", y=israel_temp,
                            color="StringYear", labels={"y": "Temperature"},
                            title="Avrage daily temperature as a function of  "
                                  "the 'DayOfYear'")
    israel_fig.show()

    israel_data_with_temp = pd.concat([israel_data, israel_temp], axis=1)
    months_groups = israel_data_with_temp.groupby("Month").agg({"Temp": "std"})
    months_fig = px.bar(months_groups, x=design_matrix[
        "Month"].drop_duplicates(), y='Temp', labels={"x": "months", "Temp":
        "standard deviation of daily temprature"},
                        title="Standard deviation of the daily temperature "
                              "as a function of months")
    months_fig.show()

    # Question 3 - Exploring differences between countries
    country_month_groups = data.groupby(["Country", "Month"])
    mean_country_month = country_month_groups.mean().reset_index()
    std_country_month = country_month_groups.std().reset_index()
    mean_country_month.insert(0, "std", std_country_month["Temp"])
    c_m_fig = px.line(mean_country_month, x='Month', y='Temp', labels={
        "Temp": "Average monthly temperatures"}, line_group='Country',
                      color='Country', error_y='std', title="Average monthly "
                        "temperature with error bars, coded by countries")
    c_m_fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_x, train_y, test_x, test_y = split_train_test(israel_data,
                                                        israel_temp, 0.75)
    losses = []
    for k in range(1, 11):
        model = PolynomialFitting(k)
        model.fit(train_x["DayOfYear"].to_numpy(), train_y.to_numpy())
        k_loss = round(model.loss(test_x["DayOfYear"].to_numpy(),
                                       test_y.to_numpy()), 2)
        losses.append(k_loss)
        print(f"test error for k={k}: {k_loss}")

    losses_fig = px.bar(losses, x=[i for i in range(1, 11)], y=losses,
                        title='Test error according to each value of k',
                        labels={'x': 'k values', 'y': 'test errors'})
    losses_fig.show()

    # Question 5 - Evaluating fitted model on different countries
    k = np.argmin(losses) + 1
    model = PolynomialFitting(k)
    model.fit(train_x["DayOfYear"].to_numpy(), train_y.to_numpy())

    countries = (design_matrix[design_matrix["Country"] != "Israel"])[
        "Country"].drop_duplicates()
    countries_losses = []
    for c in countries:
        curr_country_data = data[data['Country'] == c]
        countries_losses.append(model.loss(curr_country_data["DayOfYear"],
                                      curr_country_data["Temp"]))


    countries_losses_fig = px.bar(countries_losses, x=countries,
                                  y=countries_losses,
                                  color=countries,
                        title="Israel's model error over other countries",
                        labels={'x': 'countries', 'y': 'loss'})
    countries_losses_fig.show()