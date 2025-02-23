from data_provider.data_ml_base import (Dataset_ETT_hour_ML, Dataset_ETT_minute_ML, Dataset_Custom_ML, Dataset_Solar_ML,
                                        Dataset_AirDelay_ML)

data_dict = {
    'ETTh1': Dataset_ETT_hour_ML,
    'ETTh2': Dataset_ETT_hour_ML,
    'ETTm1': Dataset_ETT_minute_ML,
    'ETTm2': Dataset_ETT_minute_ML,
    'custom': Dataset_Custom_ML,
    'Solar': Dataset_Solar_ML,
    'AirDelay': Dataset_AirDelay_ML,
}


def data_provider(args):
    Data = data_dict[args.data]

    dataLoader = Data(
        root_path=args.root_path,
        data_name=args.data_path,
        ratio=args.ratio,
        flag='train'
    )
    train_len, test_len, train_data, test_data, df_raw, columns = dataLoader.read_data()
    print(f"train_len: {train_len}, test_len: {test_len}, train_data: {train_data.shape}, test_data: {test_data.shape}")
    return train_len, test_len, train_data, test_data, df_raw, columns
