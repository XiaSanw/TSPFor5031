
import torch
from logging import getLogger

from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from utils.utils import *


class TSPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        # === Iter-4: cosine annealing + linear warmup ===
        total_epochs = self.trainer_params['epochs']
        warmup_epochs = self.optimizer_params['scheduler'].get('warmup_epochs', 200)
        eta_min = self.optimizer_params['scheduler'].get('eta_min', 1e-6)

        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Resume: cosine-only from current epoch
            remaining = total_epochs - self.start_epoch + 1
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=remaining, eta_min=eta_min)
            self.logger.info('Saved Model Loaded !!')
        else:
            # Train from scratch: warmup + cosine
            warmup = LinearLR(self.optimizer, start_factor=0.01, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(self.optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min)
            self.scheduler = SequentialLR(self.optimizer, [warmup, cosine], milestones=[warmup_epochs])

        # === Iter-4: mixed-size training config (extended to 300) ===
        self.mixed_sizes = self.env_params.get(
            'mixed_sizes', [self.env_params['problem_size']]
        )

        # === Iter-4: Leader Reward config ===
        self.leader_reward = self.trainer_params.get('leader_reward', {})
        self.use_leader_reward = self.leader_reward.get('enable', False)
        self.leader_alpha = self.leader_reward.get('alpha', 2.0)

        # === Iter-4: training-time augmentation ===
        self.train_aug_factor = self.trainer_params.get('train_aug_factor', 1)

        # === BC -> RL transition config ===
        self.bc_transition = None  # injected by train.py

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            # === BC -> RL transition: unfreeze encoder ===
            if (self.bc_transition is not None
                    and self.bc_transition.get('freeze_encoder_epochs', 0) > 0
                    and epoch == self.bc_transition['freeze_encoder_epochs'] + 1):
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                self.optimizer.add_param_group({
                    'params': list(self.model.encoder.parameters())
                })
                self.logger.info('>>> Encoder UNFROZEN at epoch {}, params added to optimizer <<<'.format(epoch))

            self.logger.info('=================================================================')

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            # LR Decay (must be after optimizer.step() per PyTorch >= 1.1)
            self.scheduler.step()
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:  # save latest images, every epoch
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # === MODIFIED (Iter-2 patch): _train_one_batch 内部会动态缩小 batch_size ===
            # 必须取实际使用的 batch_size，否则大城市会被错误加权，且 episode 计数虚高
            avg_score, avg_loss, actual_bs = self._train_one_batch(batch_size)
            score_AM.update(avg_score, actual_bs)
            loss_AM.update(avg_loss, actual_bs)

            episode += actual_bs

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        import random

        # === Iter-4: weighted random choice of problem size ===
        # Weights biased toward large cities (pr299 is main pain point)
        problem_size = random.choices(
            self.mixed_sizes,
            weights=[1, 2, 3, 3]
        )[0]
        if problem_size != self.env.problem_size:
            self.env.problem_size = problem_size
            self.env.pomo_size = problem_size

        # === Iter-4: relaxed batch scaling (A100 40GB) ===
        aug_factor = self.train_aug_factor

        if aug_factor > 1:
            # With augmentation: effective_batch * problem_size^2 ≈ constant
            vram_budget = 32 * aug_factor * (100 ** 2)
            effective_target = int(vram_budget / (problem_size ** 2))
            effective_target = max(aug_factor * 2, min(aug_factor * 32, effective_target))
            base_batch = max(2, effective_target // aug_factor)
        else:
            # No augmentation: standard quadratic scaling, floor at 16
            base_batch = self.trainer_params['train_batch_size']
            scale = (100 / problem_size) ** 2
            base_batch = max(16, int(base_batch * scale))

        batch_size = min(base_batch, batch_size)

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size, aug_factor=aug_factor)
        # augmentation expands batch_size N-fold
        batch_size = self.env.batch_size
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # === Iter-4: Leader Reward (Wang et al., 2024) ===
        # Standard POMO baseline: per-instance mean reward
        baseline = reward.float().mean(dim=1, keepdims=True)
        advantage = reward - baseline  # shape: (batch, pomo)

        if self.use_leader_reward:
            alpha = self.leader_alpha
            # Identify leader: trajectory with highest advantage per problem
            leader_idx = advantage.argmax(dim=1)  # (batch,)
            batch_idx = torch.arange(batch_size, device=advantage.device)
            # Boost leader
            advantage[batch_idx, leader_idx] *= alpha
            # Normalize all advantages (stabilizes Adam when α > 1)
            advantage = advantage / alpha

        # REINFORCE loss
        log_prob = prob_list.log().sum(dim=2)  # (batch, pomo)
        loss = -advantage * log_prob  # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item(), batch_size

    def _train_one_epoch_bc(self, epoch, bc_data_loader):
        """
        Behavior Cloning pretraining for one epoch.

        Key design:
        1. Rotate LKH3 tour to construct expert action sequence per POMO start point
        2. Teacher forcing: env steps by expert action, model supervised at each step
        3. Loss covers all N trajectories (N=problem_size), not just trajectory 0
        """
        from TSPModel import _get_encoding

        self.model.train()
        loss_AM = AverageMeter()
        device = next(self.model.parameters()).device

        for batch_idx, batch in enumerate(bc_data_loader):
            problems = batch['problem'].to(device)   # (B, N, 2)
            tours = batch['tour'].to(device)          # (B, N)

            batch_size = problems.size(0)
            problem_size = problems.size(1)
            pomo_size = problem_size

            # ---- 1. Inject problems into env ----
            self.env.problem_size = problem_size
            self.env.pomo_size = pomo_size
            self.env.batch_size = batch_size
            self.env.problems = problems
            self.env.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, pomo_size)
            self.env.POMO_IDX = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)

            # ---- 2. Construct expert action matrix (vectorized) ----
            # tour_positions[b, city] = index of city in tours[b]
            tour_positions = torch.zeros(batch_size, problem_size, dtype=torch.long, device=device)
            positions = torch.arange(problem_size, device=device).unsqueeze(0).expand(batch_size, -1)
            tour_positions.scatter_(1, tours, positions)

            # Concatenate tour with itself to avoid modulo
            cyclic_tours = torch.cat([tours, tours], dim=1)  # (B, 2N)

            # For each start position p, expert actions are at cyclic_tours[p+1 ... p+N-1]
            offsets = torch.arange(1, problem_size, device=device)                # (N-1,)
            indices_3d = tour_positions[:, :, None] + offsets[None, None, :]      # (B, N, N-1)
            indices_flat = indices_3d.reshape(batch_size, -1)                     # (B, N*(N-1))

            expert_actions = torch.gather(cyclic_tours, 1, indices_flat)          # (B, N*(N-1))
            expert_actions = expert_actions.reshape(batch_size, pomo_size, problem_size - 1)
            # expert_actions[b, s, t] = expert action at step t+1 for trajectory s

            # ---- 3. Env reset + Encoder forward ----
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # ---- 4. POMO rollout with teacher forcing ----
            prob_list = []  # collect per-step (B, POMO) probs
            state, reward, done = self.env.pre_step()

            for step in range(problem_size):
                if step == 0:
                    # Step 0: POMO initialization — trajectory i starts from city i
                    selected = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)

                    # Set decoder q_first (first-node embedding), same as TSPModel.forward
                    encoded_first_node = _get_encoding(self.model.encoded_nodes, selected)
                    self.model.decoder.set_q1(encoded_first_node)

                    state, reward, done = self.env.step(selected)
                else:
                    # Get model's action probability distribution
                    encoded_last_node = _get_encoding(self.model.encoded_nodes, state.current_node)
                    probs = self.model.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
                    # probs: (B, POMO, N)

                    # Extract expert action probability (clamped to avoid log(0))
                    expert = expert_actions[:, :, step - 1]  # (B, POMO)
                    prob = probs[self.env.BATCH_IDX, self.env.POMO_IDX, expert]  # (B, POMO)
                    prob_list.append(prob)

                    # Teacher forcing: env steps by expert action
                    state, reward, done = self.env.step(expert)

            # ---- 5. BC Loss ----
            # prob_list has N-1 elements, each (B, POMO)
            all_probs = torch.stack(prob_list, dim=2)                # (B, POMO, N-1)
            log_probs = torch.clamp(all_probs, min=1e-8).log()      # numerical stability
            total_log_prob = log_probs.sum(dim=2)                    # (B, POMO)
            loss = -total_log_prob.mean()                            # scalar

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_AM.update(loss.item(), batch_size)

        self.logger.info('Epoch {:3d}: BC Loss: {:.4f}'.format(epoch, loss_AM.avg))
        return loss_AM.avg
