import os
from utils.tools import check_dir_exists
import numpy as np
import logging

logger = logging.getLogger('__main__')


class OutputWriter:
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting

    def write(self, data):
        folder_path = os.path.join(self.args.results, self.setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        single_metrics = np.mean(np.stack(data), axis=0).tolist()
        logger.info(f"MSE: {single_metrics[0]}, MAE: {single_metrics[1]}, RMSE: {single_metrics[2]}")

        # Write the output data to a file.
        with open(os.path.join(folder_path, 'result_long_term_forcast.txt'), 'w') as f:
            f.write(self.setting + ' \n')
            f.write(f"mse: {single_metrics[0]}, mae: {single_metrics[1]}, rmse: {single_metrics[2]}")
            f.write("\n")
            f.write("\n")
            f.close()