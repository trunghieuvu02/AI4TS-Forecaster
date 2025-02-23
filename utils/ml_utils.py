import pandas as pd
import numpy as np
import logging
from typing import List

logger = logging.getLogger('__main__')


def split_before(df: pd.DataFrame, index: int) -> (pd.DataFrame, pd.DataFrame):
    return df.iloc[:index, :], df.iloc[index:, :]


def split_numpy(data: np.ndarray, index: int) -> (np.ndarray, np.ndarray):
    return data[:index], data[index:]


def split_lens(df: pd.DataFrame, ratio: float, _border2s: List) -> (int, int):
    # border2s = [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

    # get the final border as train and test data
    data_len = _border2s[1]
    train_len = int(ratio * data_len)
    test_len = data_len - train_len

    if train_len <= 0 and test_len <= 0:
        raise ValueError("The size of training and testing data is less than or equal to 0!")

    return train_len, test_len


def print_args(args):
    logger.info("Basic Configurations:")
    logger.info(f'    {"Task Name:":<20} {args.task_name}')
    logger.info(f'    {"Model:":<20} {args.model}')
    logger.info('')

    logger.info("Data Loader: ")
    logger.info(f'    {"Data:":<20}{args.data:<20}{"Train Size:":<20}{args.ratio:<20}')
    logger.info(f'    {"Data Path":<20}{args.data_path:<20}{"Root Path:":<20}{args.root_path:<20}')

    logger.info("Model Hyperparameters:")
    logger.info(f'    {"Lags:":<20}{args.lags:<20}{"Horizon:":<20}{args.horizon:<20}')
    logger.info(
        f'    {"Normalization:":<20}{args.normalization:<20}{"Output Chunk Length:":<20}{args.output_chunk_length:<20}')
    logger.info(f'    {"Stride:":<20}{args.stride:<20}{"n_jobs:":<20}{args.n_jobs:<20}')
