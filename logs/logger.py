from lightgbm.callback import _format_eval_result
import logging
import sys
import os
import json
import argparse

if "debugpy" in sys.modules:
    current_path = '/home/nishika_apartment_2021_2'
else:
    current_path = '.'
sys.path.append(os.path.join(current_path,"logs"))
from base_log import create_logger, get_logger

jsonpath = os.path.join(current_path,"configs/default.json")

parser = argparse.ArgumentParser()
parser.add_argument('--config', default=jsonpath)
options = parser.parse_args()
config = json.load(open(options.config))

VERSION = config["exp_version"]

def log_best(model, metric):
    get_logger(VERSION).debug(model.best_iteration)
    get_logger(VERSION).debug(model.best_score['valid_0'][metric])


def log_evaluation(logger, period=1, show_stdv=True, level=logging.DEBUG):
    def _callback(env):
        if period > 0 and env.evaluation_result_list \
                and (env.iteration + 1) % period == 0:
            result = '\t'.join([
                _format_eval_result(x, show_stdv)
                for x in env.evaluation_result_list
            ])
            logger.log(level, '[{}]\t{}'.format(env.iteration + 1, result))
    _callback.order = 10
    return _callback
