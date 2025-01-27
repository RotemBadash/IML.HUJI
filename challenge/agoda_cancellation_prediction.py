from IMLearn import BaseEstimator
from challenge.agoda_cancellation_estimator import AgodaEnsembleCancellationEstimator
from challenge.agoda_cancellation_preprocessor import \
    AgodaCancellationPreprocessor

import numpy as np
import pandas as pd

WEEK = 5

def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    full_data = pd.read_csv(filename).drop_duplicates()
    full_data = full_data.drop(
        full_data[full_data["cancellation_policy_code"] == "UNKNOWN"].index)
    full_data["cancellation_datetime"].fillna(0, inplace=True)
    return full_data.dropna()


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(
        filename, index=False)


def fill_missing_columns(design_matrix, test_set):
    missing_cols = set(design_matrix.columns) - set(test_set.columns)
    for c in missing_cols:
        test_set[c] = 0
    return test_set[design_matrix.columns]


if __name__ == '__main__':
    np.random.seed(0)

    # Load and preprocess data
    full_data = load_data(
        "../datasets/agoda_cancellation_train.csv")
    p = AgodaCancellationPreprocessor(full_data)
    design_matrix = p.preprocess(full_data)
    cancellation_labels = p.preprocess_labels(full_data.cancellation_datetime,
                                              full_data.booking_datetime)
    for i in range(1, WEEK):
        test_set_i = p.preprocess(pd.read_csv(f"test_set_week_{i}.csv"))
        test_set_i_labels = pd.read_csv(f"test_set_week_{i}_labels.csv")[
            "h_booking_id|label"].astype(str).apply(lambda x: int(x[-1]))
        test_set_i = fill_missing_columns(design_matrix, test_set_i)
        design_matrix = pd.concat([design_matrix, test_set_i])
        cancellation_labels = pd.concat([cancellation_labels, test_set_i_labels])

    # Fit model over data
    estimator = AgodaEnsembleCancellationEstimator() \
        .fit(design_matrix, cancellation_labels)

    test_set = p.preprocess(pd.read_csv(f"test_set_week_{WEEK}.csv"))
    # Expand test_set with 0 columns to fit the design_matrix shape
    test_set = fill_missing_columns(design_matrix, test_set)

    # Store model predictions over test set
    evaluate_and_export(estimator, test_set,
                        "319091385_314618794_318839610.csv")
