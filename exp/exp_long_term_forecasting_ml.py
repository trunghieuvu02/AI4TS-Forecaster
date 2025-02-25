import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler
from exp.exp_basic_ml_models import ModelBase
from darts import TimeSeries
import argparse
import logging
import warnings
from utils.metrics import get_ml_metric
from utils.ml_utils import split_numpy
from data_provider.data_ml_factory import data_provider
from utils.output_writer import OutputWriter
import time

logger = logging.getLogger('__main__')
warnings.filterwarnings("ignore")


# This class is used for long-term forecasting using machine learning models.
class ML_Long_Term_Forecast(ModelBase):
    def __init__(self, args: argparse.Namespace):
        """
        Initialize the ML_Long_Term_Forecast class.

        Args:
            args (argparse.Namespace): The arguments for the model.
        """
        super().__init__(args)
        # self.df_raw = None
        # self.scaler = StandardScaler()

        self.all_test_results = []
        self.all_rolling_actual = []
        self.all_rolling_predict = []
        self.testing_time = None

        if self.args.model_type == "STATS":
            self.model = self.get_stas_model(self.args.model)
        elif self.args.model_type == "ML":
            self.model = self.get_ml_model(self.args.model)

    def runner(self, setting: str) -> None:
        """
        Train the model with the training data and evaluate it with the testing data.
        """
        training_time = time.time()
        logger.info("Start training model...")
        train_len, test_len, train_data, test_data, data, columns = data_provider(self.args)
        self.model = self._fit_model(train_data, columns)
        logger.info(f"Model training completed - Take: {time.time() - training_time:.2f}s")

        index_list = self.get_index(train_len, test_len, horizon=self.args.horizon, stride=self.args.stride)

        logger.info("Start evaluating model...")
        for index in tqdm(index_list):
            self.testing_time = time.time()
            train, rest = split_numpy(data, index)
            test, _ = split_numpy(rest, self.args.horizon)

            predict = self._forecast(train, columns)

            mse, mae, rmse = get_ml_metric(true=test, pred=predict)
            single_metric = [mse, mae, rmse]

            inference_data = pd.DataFrame(
                predict, columns=columns
            )

            self.all_rolling_actual.append(test)
            self.all_rolling_actual.append(inference_data)
            self.all_test_results.append(single_metric)

        output_writer = OutputWriter(self.args, setting)
        output_writer.write(self.all_test_results)

    def _fit_model(self, train_data: pd.DataFrame, columns):
        """
        Fit the model with the training data.

        Args:
            train_data (pd.DataFrame): The training data for the model.
        """
        # if self.args.normalization:
        #     self.scaler.fit(train_data.values)
        #     train_data = pd.DataFrame(
        #         self.scaler.transform(train_data.values),
        #         columns=train_data.columns,
        #         index=train_data.index
        #     )

        train_data = pd.DataFrame(
            train_data,
            columns=columns,
        )

        train_data = TimeSeries.from_dataframe(train_data)

        self.model.fit(train_data)

        return self.model

    def _forecast(self, series: pd.DataFrame, columns) -> np.ndarray:
        """
        Forecast the future values based on the input series.

        Args:
            series (pd.DataFrame): The input series for forecasting.

        Returns:
            np.ndarray: The forecasted values.
        """
        # if self.args.normalization:
        #     series = pd.DataFrame(
        #         self.scaler.transform(series.values),
        #         columns=series.columns,
        #         index=series.index
        #     )

        series = pd.DataFrame(
            series,
            columns=columns,
        )
        series = TimeSeries.from_dataframe(series)
        # print("==============> series: ", len(series))
        # print("==============> horizon: ", self.args.horizon)
        predict_time = time.time()
        results = self.model.predict(self.args.horizon, series)
        predict = results.values()
        # if self.args.normalization:
        #     predict = self.scaler.inverse_transform(predict)

        return predict

    @staticmethod
    def get_index(train_len: int, test_len: int, horizon: int, stride: int) -> list:
        """
         Get the index list for the data.

         Args:
             train_len (int): The length of the training data.
             test_len (int): The length of the testing data.
             horizon (int): The forecasting horizon.
             stride (int): The stride for the index.

         Returns:
             list: The index list.
         """
        data_len = train_len + test_len
        # why the index_list is generated in this way?
        # data_len - horizon + 1: the number of the data points that can be used for forecasting
        index_list = list(range(train_len, data_len - horizon + 1, stride))
        if (test_len - horizon) % stride != 0:
            index_list.append(data_len - horizon)
        return index_list
