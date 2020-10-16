import math
import numpy as np

#RMSPE
def root_mean_squared_percentage_error(y_true, prediction):
  y_true, y_pred = np.array(y_true), np.array(prediction)
  EPSILON =  1e-10
  rmspe = (np.sqrt(np.mean(np.square((y_true - y_pred) / (y_true + EPSILON))))) * 100
  return rmspe

#Define MAPE function
def mean_absolute_percentage_error(y_true, prediction):
    y_true, y_pred = np.array(y_true), np.array(prediction)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape