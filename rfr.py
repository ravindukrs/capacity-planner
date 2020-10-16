#Imports
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from metric_functions import root_mean_squared_percentage_error, mean_absolute_percentage_error


def run_rfr(label):
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

        # Folds
        kf = KFold(n_splits=10)

        predictions = []
        y_actual = []
        for train_index, test_index in kf.split(X):
            gsc = GridSearchCV(
                estimator=RandomForestRegressor(),
                param_grid={
                    'max_depth': (3, 5, 7),
                    'n_estimators': (100, 1000),
                },
                cv=5,
                scoring='neg_mean_squared_error',
                verbose=0,
                n_jobs=-1
            )

            print("\Predicting Using: TEST INDEX:", test_index)
            grid_result = gsc.fit(X[train_index], Y[train_index])
            best_params = grid_result.best_params_
            print("\nBest: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

            rfr = RandomForestRegressor(
                max_depth=best_params["max_depth"],
                n_estimators=best_params["n_estimators"],
                random_state=best_params["random_state"],
                verbose=False
            )

            rfr.fit(X[train_index], Y[train_index])

            y_pred = rfr.predict(X[test_index])

            for item in y_pred:
                predictions.append(item)
            for item in Y[test_index]:
                y_actual.append(item)

        RMSPE = root_mean_squared_percentage_error(y_actual, predictions)
        MAPE = mean_absolute_percentage_error(y_actual, predictions)
        RMSE = math.sqrt(mean_squared_error(y_actual, predictions))

        print(
            "Scores for RFR: " + label + "\n",
            "RMSE :", RMSE, "\n",
            "MAPE: ", MAPE, "\n",
            "RMSPE: ", RMSPE, "\n",
        )

        file_name = "/results/" + "rfr_"+label+ ".csv"
        with open(file_name, "a") as f:
            writer = csv.writer(f)
            writer.writerows(zip(y_actual, predictions))
    else:
        print("Invalid Parameters for RFR Run Function")