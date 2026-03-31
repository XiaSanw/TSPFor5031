##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 3


##########################################################################################
# Path Config

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# POMO/NEW_py_ver utils
sys.path.insert(0, "../..")  # for utils

# TSProblemDef (used by augmentation)
sys.path.insert(0, "..")  # for TSProblemDef

# Local TSPLIB dataset (bundled with POMO)
TSP_DATA_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


##########################################################################################
# import

import time
import logging
from datetime import datetime

import pytz

from utils.utils import create_logger, copy_all_src, get_result_folder

from TSPTester_LIB import TSPTester_LIB


##########################################################################################
# parameters
# need to be modified

augmentation_enable = True
# if True, per-instance lists will be dumped at the end.
detailed_log = True

lib_path = os.path.join(TSP_DATA_ROOT, "data", "val")

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_tsp100_model2_longTrain',
        'epoch': 3000,
    },
    'filename': lib_path,
    'augmentation_enable': augmentation_enable,
    'aug_factor': 8,
    'detailed_log': detailed_log,
    # Only EUC_2D / CEIL_2D are supported (same as ICAM's LIBUtils.TSPLIBReader)
    'scale_range_all': [[0, 10000]],
}

if tester_params['augmentation_enable']:
    highlight = f'aug{tester_params["aug_factor"]}'
else:
    highlight = 'no_aug'

process_start_time = datetime.now(pytz.timezone("Asia/Shanghai"))
logger_params = {
    'log_file': {
        'desc': f'{highlight}_test_TSPLIB_POMO',
        'filename': 'run_log.txt',
        'filepath': './result_lib/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}',
    }
}


##########################################################################################
##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = TSPTester_LIB(model_params=model_params, tester_params=tester_params)

    # copy source snapshot
    copy_all_src(get_result_folder())

    tester.run_lib()


def _set_debug_mode():
    global tester_params
    tester_params['scale_range_all'] = [[0, 100]]


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
