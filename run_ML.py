import argparse
import os
from exp.exp_long_term_forecasting_ml import ML_Long_Term_Forecast
import logging
from utils.tools import check_dir_exists
from utils.ml_utils import print_args

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("TimeSeries-AdvancedMethods-Hub")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Machine Learning setting')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
    parser.add_argument('--model', type=str, default='LinearRegressionModel', help='model name')
    parser.add_argument('--model_type', type=str, default='ML', help='model type',
                        choices=['ML', 'STATS'])
    # data loader
    parser.add_argument('--root_path', type=str, default='./dataset/ETT-small')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--data', type=str, default='ETTh1')
    parser.add_argument('--ratio', type=float, default=0.8)

    # model hyperparameters
    parser.add_argument('--lags', type=int, default=96)
    parser.add_argument('--output_chunk_length', type=int, default=1)
    parser.add_argument('--normalization', type=bool, default=False)
    parser.add_argument('--horizon', type=int, default=96)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--n_jobs', type=int, default=12)

    # logger dir
    parser.add_argument('--logger_dir', type=str, default='./logger/', help='logger dir')

    # output dir
    parser.add_argument('--results', type=str, default='./results/ML_results', help='output dir')
    args = parser.parse_args()

    # add logger files
    check_dir_exists(args.logger_dir)
    file_handler = logging.FileHandler(os.path.join(args.logger_dir, 'Stats_logger.log'))
    logger.addHandler(file_handler)

    logger.info('Args in experiment:')
    print_args(args)

    Exp = ML_Long_Term_Forecast(args)

    if args.task_name == 'long_term_forecast':
        settings = '{}_{}_{}_{}_lags{}_horizon{}_ratio{}_modeltype{}'.format(
            args.task_name,
            args.data_path.split(".")[0],
            args.model,
            args.data,
            args.lags,
            args.horizon,
            args.ratio,
            args.model_type
        )

    Exp.runner(settings)
