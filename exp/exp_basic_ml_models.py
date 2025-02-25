import numpy as np
import pandas as pd
from darts.models import LinearRegressionModel, XGBModel, RegressionModel, LightGBMModel, RandomForest
from darts.models import VARIMA, KalmanForecaster
from statsmodels.tsa.vector_ar.var_model import VAR
import functools
import argparse


class ModelBase(object):
    def __init__(self, args: argparse.Namespace):
        # self.model_dict = None
        self.args = args

        # ML model parameters
        self.model_ml_paramters_args = {
            "lags": self.args.lags,
            "output_chunk_length": self.args.output_chunk_length,
            "n_jobs": self.args.n_jobs
        }

        # Statistical model parameters
        self.model_stat_paramters_args = {
            # "normalization": self.args.normalization,
        }

    def get_ml_model(self, model_name: str) -> "ModelBase":
        model_dict = {
            "LinearRegressionModel": LinearRegressionModel,
            "LightGBMModel": LightGBMModel,
            "XGBModel": XGBModel,
            "RandomForestModel": RandomForest,
        }

        return functools.partial(model_dict[model_name], **self.model_ml_paramters_args)()

    def get_stas_model(self, model_name: str) -> "ModelBase":
        model_dict = {
            "VAR": VAR,
            "VARIMA": VARIMA,
            "KalmanForecaster": KalmanForecaster,
        }

        return functools.partial(model_dict[model_name], **self.model_stat_paramters_args)()

    def fit_forecasting(self):
        pass

    def forecast(self):
        pass

    def train(self):
        pass
