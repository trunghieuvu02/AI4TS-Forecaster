import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    return mse, mae, rmse, mape, mspe


def metric_ml(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    return mse, mae, rmse


def _error_norm(actual, predicted, scaler):
    return scaler.transform(actual) - scaler.transform(predicted)


def get_mse_norm(actual, predicted, scaler):
    return np.mean(np.square(_error_norm(actual, predicted, scaler)))


def get_rmse_norm(actual, predicted, scaler):
    return np.sqrt(get_mse_norm(actual, predicted, scaler))


def get_mae_norm(actual, predicted, scaler):
    return np.mean(np.abs(_error_norm(actual, predicted, scaler)))


def get_ml_metric(true, pred):
    # mse = get_mse_norm(true, pred, scaler)
    # mae = get_mae_norm(true, pred, scaler)
    # rmse = get_rmse_norm(true, pred, scaler)
    mse, mae, rmse = metric_ml(true, pred)
    return mse, mae, rmse
