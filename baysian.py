import math
import numpy as np
import pandas as pd
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, KFold
import pymc3 as pm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from metric_functions import root_mean_squared_percentage_error, mean_absolute_percentage_error


class BayesianRBFRegression:
    def fit(self, X, Y):
        with pm.Model() as self.model:
            ℓ = pm.Gamma("ℓ", alpha=1, beta=1)
            η = pm.HalfCauchy("η", beta=5)

            cov = η ** 2 * pm.gp.cov.Matern52(X.shape[1], ℓ)
            self.gp = pm.gp.Marginal(cov_func=cov)

            σ = pm.HalfCauchy("σ", beta=0.1)  # 5, 0.5, 2
            y_ = self.gp.marginal_likelihood("y", X=X, y=Y, noise=σ)

            self.map_trace = [pm.find_MAP()]

    def predict(self, X, with_error=False):
        with self.model:
            f_pred = self.gp.conditional('f_pred', X)
            pred_samples = pm.sample_ppc(self.map_trace, vars=[f_pred], samples=2000)
            y_pred, uncer = pred_samples['f_pred'].mean(axis=0), pred_samples['f_pred'].std(axis=0)

        if with_error:
            return y_pred, uncer / 1000
        return y_pred


def eval_bayesian_rbf(X, y, eval_X, eval_y):
    lr = BayesianRBFRegression()
    lr.fit(X, y)
    pred_y, error = lr.predict(eval_X, True)

    return pred_y, error


def run_baysian(label):
    predict_label = None
    if label == "tps":
        predict_label = 9
    elif label == "latency":
        predict_label = 10

    if predict_label == 9 or predict_label == 10:
        # Read Data
        dataset = pd.read_csv('dataset/dataset.csv')

        # Ignore Errors
        dataset = dataset.loc[dataset["Error %"] < 5]

        # Define X and Y columns
        X = dataset.iloc[:, [0, 2, 3]].values
        Y = dataset.iloc[:, predict_label].values  # 10 for Latancy, 10 for TP

        # Encode 'Scenario Name'
        le_X_0 = LabelEncoder()
        X[:, 0] = le_X_0.fit_transform(X[:, 0])

        # Create Scaler
        scaler = MinMaxScaler(feature_range=(0, 1))

        # Apply Scaler on X
        scaler.fit(X)

        # Convert Y to 1D Array - Not necessary
        Y = Y.flatten()

        # Shuffle Data
        X, Y = shuffle(X, Y, random_state=42)

        predictions = []
        errorlist = []
        y_actual = []

        kf = KFold(n_splits=10)

        for train_index, test_index in kf.split(X):
            pred_bayes, error = eval_bayesian_rbf(np.copy(X[train_index]), np.copy(Y[train_index]),
                                                  np.copy(X[test_index]), np.copy(Y[test_index]));

            for item in pred_bayes:
                predictions.append(item)

            for item in error:
                errorlist.append(item)

            for item in Y[test_index]:
                y_actual.append(item)

        RMSPE = root_mean_squared_percentage_error(y_actual, predictions)
        MAPE = mean_absolute_percentage_error(y_actual, predictions)
        RMSE = math.sqrt(mean_squared_error(y_actual, predictions, squared=True))

        print(
            "Scores for Baysian_Matern: " + label + "\n",
            "RMSE :", RMSE, "\n",
            "MAPE: ", MAPE, "\n",
            "RMSPE: ", RMSPE, "\n",
        )

        file_name = "/results/" + "baysian_" + label + ".csv"
        with open(file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerows(zip(y_actual, predictions))
    else:
        print("Invalid Parameters for Baysian Run Function")
