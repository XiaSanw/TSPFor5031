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
import torch
import random as _random
from utils.utils import create_logger, copy_all_src

from TSPTrainer import TSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    # === Iter-4: extended to 250 cities; 300 removed (OOM with scaled model) ===
    'mixed_sizes': [100, 150, 200, 250],
}

model_params = {
    # === Iter-4: scaled-up model — 1.6M → ~3.5M params ===
    # emb=256/FF=1024/9layers OOM on POMO 300-step rollout even on A100 40GB.
    # Compromise: 1.5-2.25x baseline, fits VRAM with batch=32.
    'embedding_dim': 192,
    'sqrt_embedding_dim': 192**(1/2),
    'encoder_layer_num': 8,
    'qkv_dim': 16,
    'head_num': 12,
    'logit_clipping': 10,
    'ff_hidden_dim': 768,
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
    'train_batch_size': 32,  # Iter-4: halved for scaled model VRAM
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
    # Disabled initially — scaled model + aug8 OOM even on A100 40GB.
    # Re-enable after verifying base training VRAM is stable.
    'train_aug_factor': 1,
}

# ========== BC Pretraining Config ==========
BC_CONFIG = {
    'enable': True,
    'data_paths': [
        './lkh3_expert_data/n100/lkh3_data_n100.pt',
        './lkh3_expert_data/n150/lkh3_data_n150.pt',
        './lkh3_expert_data/n200/lkh3_data_n200.pt',
    ],
    'epochs': 200,           # BC training epochs
    'batch_size': 64,        # BC batch size (teacher forcing saves memory)
    'lr': 1e-4,              # BC learning rate (smaller than RL)
    # BC -> RL transition
    'freeze_encoder_epochs': 100,   # freeze encoder for first N RL epochs
    'rl_warmup_epochs': 50,         # RL lr linear warmup duration
    'rl_warmup_lr_start': 1e-5,     # warmup starting lr
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

    # =================================================================
    # Phase 1: Behavior Cloning Pretraining
    # =================================================================
    if BC_CONFIG['enable']:
        from bc_dataset import build_bc_loaders

        logger = logging.getLogger('root')
        logger.info("=" * 60)
        logger.info("Phase 1: Behavior Cloning Pretraining")
        logger.info("=" * 60)

        # Per-size independent DataLoaders (different shapes can't mix in one batch)
        bc_loaders = build_bc_loaders(
            BC_CONFIG['data_paths'],
            batch_size=BC_CONFIG['batch_size'],
            num_workers=2
        )

        # BC uses independent optimizer to avoid polluting RL Adam momentum
        bc_optimizer = torch.optim.Adam(
            trainer.model.parameters(),
            lr=BC_CONFIG['lr'],
            weight_decay=optimizer_params['optimizer'].get('weight_decay', 1e-6)
        )
        original_optimizer = trainer.optimizer
        trainer.optimizer = bc_optimizer

        # BC training loop: random problem size per epoch
        for bc_epoch in range(1, BC_CONFIG['epochs'] + 1):
            size = _random.choice(list(bc_loaders.keys()))
            bc_loss = trainer._train_one_epoch_bc(bc_epoch, bc_loaders[size])
            if bc_epoch % 50 == 0 or bc_epoch == 1:
                logger.info('BC Epoch {:3d}/{:3d} (n={}): Loss = {:.4f}'.format(
                    bc_epoch, BC_CONFIG['epochs'], size, bc_loss))

        # Save BC pretrained weights
        bc_ckpt_path = '{}/bc_pretrained.pt'.format(trainer.result_folder)
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
        }, bc_ckpt_path)
        logger.info('BC Pretraining Done! Saved to {}'.format(bc_ckpt_path))

        # =============================================================
        # BC -> RL Transition Setup
        # =============================================================
        # 1. Freeze encoder (prevent RL policy drift from destroying BC representations)
        if BC_CONFIG['freeze_encoder_epochs'] > 0:
            for param in trainer.model.encoder.parameters():
                param.requires_grad = False
            logger.info('Encoder frozen for first {} RL epochs'.format(
                BC_CONFIG['freeze_encoder_epochs']))

        # 2. Rebuild optimizer for RL (only trainable params)
        #    base_lr = target lr (3e-4); LinearLR start_factor controls warmup start
        #    epoch 0: lr = 3e-4 * (1e-5/3e-4) = 1e-5; after warmup: lr = 3e-4
        rl_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, trainer.model.parameters()),
            lr=optimizer_params['optimizer']['lr'],
            weight_decay=optimizer_params['optimizer'].get('weight_decay', 1e-6)
        )
        trainer.optimizer = rl_optimizer
        logger.info('RL optimizer rebuilt, base_lr = {}, warmup from {}'.format(
            optimizer_params['optimizer']['lr'], BC_CONFIG['rl_warmup_lr_start']))

        # 3. Rebuild scheduler (warmup + cosine, replacing original scheduler)
        total_rl_epochs = trainer_params['epochs']
        warmup_epochs = BC_CONFIG['rl_warmup_epochs']
        eta_min = optimizer_params['scheduler'].get('eta_min', 1e-6)

        from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
        rl_warmup = LinearLR(
            rl_optimizer,
            start_factor=BC_CONFIG['rl_warmup_lr_start'] / optimizer_params['optimizer']['lr'],
            end_factor=1.0,
            total_iters=warmup_epochs
        )
        rl_cosine = CosineAnnealingLR(
            rl_optimizer, T_max=total_rl_epochs - warmup_epochs, eta_min=eta_min
        )
        trainer.scheduler = SequentialLR(
            rl_optimizer, [rl_warmup, rl_cosine], milestones=[warmup_epochs]
        )
        logger.info('RL scheduler rebuilt: warmup {} epochs -> cosine decay'.format(
            warmup_epochs))

        # Inject transition config into trainer (for encoder unfreeze)
        trainer.bc_transition = {
            'freeze_encoder_epochs': BC_CONFIG['freeze_encoder_epochs'],
        }

    # =================================================================
    # Phase 2: REINFORCE Fine-tuning
    # =================================================================
    logger = logging.getLogger('root')
    logger.info("=" * 60)
    logger.info("Phase 2: RL Fine-tuning")
    logger.info("=" * 60)
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
