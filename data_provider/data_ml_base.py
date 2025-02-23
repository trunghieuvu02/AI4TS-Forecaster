import os
import numpy as np
import pandas as pd
from utils.ml_utils import split_before, split_lens, split_numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")


class Dataset_ETT_hour_ML:
    def __init__(self, root_path: str, data_name: str, ratio, scale=True, flag='train'):
        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.ratio = ratio
        self.scale = scale

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

    def read_data(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_name))
        df_raw['date'] = pd.to_datetime(df_raw.date)
        df_raw = df_raw.set_index('date')

        _border2s = [12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]

        train_len, test_len = split_lens(df_raw, self.ratio, _border2s)

        if self.scale:
            train_data = df_raw.iloc[:train_len, :]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        columns = df_raw.columns
        # train_data, test_data = split_before(df_raw, train_len)
        train_data, test_data = split_numpy(data, train_len)

        return train_len, test_len, train_data, test_data, data, columns


class Dataset_ETT_minute_ML:
    def __init__(self, root_path: str, data_name: str, ratio, scale=True, flag='train'):
        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.ratio = ratio
        self.scale = scale

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

    def read_data(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_name))
        df_raw['date'] = pd.to_datetime(df_raw.date)
        df_raw = df_raw.set_index('date')

        _border2s = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        train_len, test_len = split_lens(df_raw, self.ratio, _border2s)

        if self.scale:
            train_data = df_raw.iloc[:train_len, :]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        columns = df_raw.columns
        # train_data, test_data = split_before(df_raw, train_len)
        train_data, test_data = split_numpy(data, train_len)

        return train_len, test_len, train_data, test_data, data, columns


class Dataset_Custom_ML:
    """
    Map-stype datasets
    small dataset -> storage/ load into RAM
    cannot apply for the big dataset
    """

    def __init__(self, root_path: str, data_name: str, ratio, scale=True, flag='train'):
        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.ratio = ratio
        self.scale = scale

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

    def read_data(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_name))
        df_raw['date'] = pd.to_datetime(df_raw.date)
        df_raw = df_raw.set_index('date')

        num_test = int(len(df_raw) * 0.2)
        num_train = len(df_raw) - num_test

        _border2s = [num_train, len(df_raw)]

        if self.scale:
            train_data = df_raw.iloc[:num_train, :]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        columns = df_raw.columns
        # train_len, test_len = split_lens(df_raw, self.ratio, _border2s)
        # train_data, test_data = data[:num_train, :], data[num_train:, :]
        train_data, test_data = split_numpy(data, num_train)
        return num_train, num_test, train_data, test_data, data, columns


class Dataset_Solar_ML:
    def __init__(self, root_path: str, data_name: str, ratio, scale=True, flag='train'):
        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.ratio = ratio
        self.scale = scale

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

    def read_data(self):
        self.scaler = StandardScaler()

        df_raw = []
        with open(os.path.join(self.root_path, self.data_name), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)

        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_test = int(len(df_raw) * 0.2)
        num_train = len(df_raw) - num_test

        _border2s = [num_train, len(df_raw)]

        if self.scale:
            train_data = df_raw.iloc[:num_train, :]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        columns = df_raw.columns
        train_data, test_data = split_numpy(data, num_train)
        return num_train, num_test, train_data, test_data, data, columns


class Dataset_AirDelay_ML:
    def __init__(self, root_path: str, data_name: str, ratio, scale=True, flag='train'):
        self.root_path = root_path
        self.data_name = data_name
        self.flag = flag
        self.ratio = ratio
        self.scale = scale

        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]

    def read_data(self):
        self.scaler = StandardScaler()

        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_name))

        df_raw = df_raw.dropna()
        df_raw = df_raw.reset_index(drop=True)

        df_raw['FlightDate'] = pd.to_datetime(df_raw['FlightDate'])
        df_raw = df_raw.set_index('FlightDate')

        num_test = int(len(df_raw) * 0.2)
        num_train = len(df_raw) - num_test

        _border2s = [num_train, len(df_raw)]

        if self.scale:
            train_data = df_raw.iloc[:num_train, :]
            self.scaler.fit(train_data.values)

            data = self.scaler.transform(df_raw.values)
        else:
            data = df_raw.values

        columns = df_raw.columns
        # train_len, test_len = split_lens(df_raw, self.ratio, _border2s)
        # train_data, test_data = data[:num_train, :], data[num_train:, :]
        train_data, test_data = split_numpy(data, num_train)
        return num_train, num_test, train_data, test_data, data, columns








