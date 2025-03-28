# metrics.py

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred, squared=False)
