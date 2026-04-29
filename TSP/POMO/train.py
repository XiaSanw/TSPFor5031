##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTrainer import TSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    # === Iter-4: extended to 300 cities (A100 40GB) ===
    'mixed_sizes': [100, 150, 200, 250, 300],
}

model_params = {
    # === Iter-4: scaled-up model (A100 40GB) — 1.6M → 6.8M params ===
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 9,
    'qkv_dim': 16,
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 1024,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 3e-4,
        'weight_decay': 1e-6
    },
    # === Iter-4: cosine annealing + linear warmup (replaces MultiStepLR) ===
    'scheduler': {
        'warmup_epochs': 200,
        'eta_min': 1e-6,
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    # === Iter-4: train from scratch with scaled-up model on A100 ===
    'epochs': 5000,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'logging': {
        'model_save_interval': 100,
        'img_save_interval': 100,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        # === Iter-4: train from scratch (architecture changed) ===
        'enable': False,
        'path': '',
        'epoch': 0,
    },
    # === Iter-4: Leader Reward (Wang et al., 2024) ===
    'leader_reward': {
        'enable': True,
        'alpha': 2.0,
    },
    # === Iter-4: training-time instance augmentation ===
    'train_aug_factor': 8,
}

logger_params = {
    'log_file': {
        'desc': 'train__tsp_mixed100-300__scaledModel__leaderReward',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 10
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
