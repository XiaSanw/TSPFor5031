import os
import time
from dataclasses import dataclass
from logging import getLogger
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from TSPEnv import TSPEnv as Env, Step_State
from TSPModel import TSPModel as Model

from TSProblemDef import augment_xy_data_by_8_fold

from tsplib_utils import TSPLIBReader, tsplib_cost


def _normalize_to_unit_square(node_xy: torch.Tensor) -> torch.Tensor:
    """Normalize to [0,1] with uniform scaling (same style as ICAM script)."""
    xy_max = torch.max(node_xy, dim=1, keepdim=True).values
    xy_min = torch.min(node_xy, dim=1, keepdim=True).values
    ratio = torch.max((xy_max - xy_min), dim=-1, keepdim=True).values
    ratio[ratio == 0] = 1
    return (node_xy - xy_min) / ratio.expand(-1, 1, 2)

###############################################################################

def _compute_dist_matrix(coords, edge_weight_type='EUC_2D'):
    """计算距离矩阵。coords: (N, 2) tensor -> list of list (Python 嵌套列表)"""
    diff = coords.unsqueeze(0) - coords.unsqueeze(1)
    dists = (diff ** 2).sum(dim=2).sqrt()
    if edge_weight_type == 'CEIL_2D':
        dists = torch.ceil(dists)
    elif edge_weight_type == 'EUC_2D':
        dists = torch.floor(dists + 0.5)
    return dists.cpu().tolist()


def _two_opt(tour, dist_matrix, max_iter=10000):
    """
    经典 2-opt 局部搜索。
    tour: 城市索引列表，如 [0, 3, 1, 2]
    dist_matrix: Python 嵌套列表，dist_matrix[i][j] = 城市 i 到 j 的距离
    返回优化后的 tour。
    """
    n = len(tour)
    tour = list(tour)
    improved = True
    iter_count = 0
    while improved and iter_count < max_iter:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                # 当前边：tour[i-1]->tour[i] 和 tour[j]->tour[j+1]
                # 候选边：tour[i-1]->tour[j] 和 tour[i]->tour[j+1]
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (dist_matrix[a][c] + dist_matrix[b][d]
                            - dist_matrix[a][b] - dist_matrix[c][d])
                if delta < -1e-6:
                    # 反转 tour[i..j] 这一段
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
                    break
            if improved:
                break
        iter_count += 1
    return tour

def _tour_length(tour, dist_matrix):
    """计算一条路线的总长度。"""
    n = len(tour)
    return sum(dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n))


###############################################################################


@dataclass
class LibResult:
    instances: List[str]
    optimal: List[Optional[float]]
    problem_size: List[int]
    no_aug_score: List[float]
    aug_score: List[float]
    no_aug_gap: List[Optional[float]]
    aug_gap: List[Optional[float]]
    total_instance_num: int = 0
    solved_instance_num: int = 0

    @staticmethod
    def _mean_valid(values: List[Optional[float]]) -> Optional[float]:
        valid_values = [value for value in values if value is not None]
        if not valid_values:
            return None
        return float(np.mean(valid_values))

    @property
    def avg_no_aug_gap(self) -> Optional[float]:
        return self._mean_valid(self.no_aug_gap)

    @property
    def avg_aug_gap(self) -> Optional[float]:
        return self._mean_valid(self.aug_gap)

    def to_dict(self) -> Dict[str, object]:
        return {
            "instances": self.instances,
            "optimal": self.optimal,
            "problem_size": self.problem_size,
            "no_aug_score": self.no_aug_score,
            "aug_score": self.aug_score,
            "no_aug_gap": self.no_aug_gap,
            "aug_gap": self.aug_gap,
            "total_instance_num": self.total_instance_num,
            "solved_instance_num": self.solved_instance_num,
            "avg_no_aug_gap": self.avg_no_aug_gap,
            "avg_aug_gap": self.avg_aug_gap,
        }


class TSPTester_LIB:
    def __init__(self, model_params, tester_params):
        self.model_params = model_params
        self.tester_params = tester_params

        self.logger = getLogger('root')

        use_cuda = self.tester_params['use_cuda']
        if use_cuda:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        self.model = Model(**self.model_params)

        checkpoint_fullname = tester_params.get('checkpoint_path')
        if checkpoint_fullname is None:
            model_load = tester_params['model_load']
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        total = sum([param.nelement() for param in self.model.parameters()])
        self.logger.info("Model loaded from: {}".format(checkpoint_fullname))
        self.logger.info("Number of parameters: %.2fM" % (total / 1e6))

    def run_lib(self) -> LibResult:
        filename = self.tester_params['filename']
        scale_range_all = self.tester_params.get('scale_range_all', [[0, 1000]])
        detailed_log = self.tester_params.get('detailed_log', False)

        start_time_all = time.time()
        all_instance_num = 0
        solved_instance_num = 0

        result = LibResult(
            instances=[],
            optimal=[],
            problem_size=[],
            no_aug_score=[],
            aug_score=[],
            no_aug_gap=[],
            aug_gap=[],
        )

        for scale_range in scale_range_all:
            self.logger.info("#################  Test scale range: {}  #################".format(scale_range))

            for root, _, files in os.walk(filename):
                for file in files:
                    if not file.endswith('.tsp'):
                        continue

                    full_path = os.path.join(root, file)
                    name, dimension, locs, ew_type = TSPLIBReader(full_path)

                    all_instance_num += 1

                    if name is None:
                        self.logger.info(f"Skip (unsupported or invalid TSPLIB): {full_path}")
                        continue

                    if not (scale_range[0] <= dimension < scale_range[1]):
                        continue

                    optimal = tsplib_cost.get(name, None)
                    if optimal is None:
                        self.logger.info(
                            f"Optimal not found for {name}. "
                            "Will report scores but leave gap fields empty."
                        )

                    self.logger.info("===============================================================")
                    self.logger.info("Instance name: {}, problem_size: {}, EDGE_WEIGHT_TYPE: {}".format(name, dimension, ew_type))

                    coords_orig_np = np.array(locs, dtype=np.float32)
                    coords_orig = torch.from_numpy(coords_orig_np).to(self.device)
                    node_coord = coords_orig[None, :, :]

                    nodes_xy_normalized = _normalize_to_unit_square(node_coord)

                    try:
                        no_aug_score, aug_score = self._test_one_instance(
                            nodes_xy_normalized=nodes_xy_normalized,
                            coords_orig=coords_orig,
                            ew_type=ew_type,
                        )
                    except Exception as e:
                        self.logger.exception(f"Failed on instance {name}: {e}")
                        continue

                    solved_instance_num += 1

                    if optimal is None:
                        no_aug_gap = None
                        aug_gap = None
                    else:
                        no_aug_gap = (no_aug_score - optimal) / optimal * 100
                        aug_gap = (aug_score - optimal) / optimal * 100

                    result.instances.append(name)
                    result.optimal.append(float(optimal) if optimal is not None else None)
                    result.problem_size.append(int(dimension))
                    result.no_aug_score.append(float(no_aug_score))
                    result.aug_score.append(float(aug_score))
                    result.no_aug_gap.append(float(no_aug_gap) if no_aug_gap is not None else None)
                    result.aug_gap.append(float(aug_gap) if aug_gap is not None else None)

                    if optimal is None:
                        self.logger.info(
                            "no public optimum. no_aug: {:.3f}, aug: {:.3f}".format(
                                no_aug_score, aug_score
                            )
                        )
                    else:
                        self.logger.info(
                            "optimal: {:.3f}, no_aug: {:.3f} (gap {:.3f}%), aug: {:.3f} (gap {:.3f}%)".format(
                                optimal, no_aug_score, no_aug_gap, aug_score, aug_gap
                            )
                        )

        end_time_all = time.time()
        result.total_instance_num = all_instance_num
        result.solved_instance_num = solved_instance_num

        self.logger.info("=========================== Summary ===========================")
        self.logger.info(
            "All done, solved instance number: {}/{}, total time: {:.2f}s".format(
                solved_instance_num, all_instance_num, end_time_all - start_time_all
            )
        )

        if solved_instance_num > 0 and result.avg_aug_gap is not None:
            self.logger.info(
                "Avg gap(no aug): {:.3f}%, Avg gap(aug): {:.3f}%".format(
                    result.avg_no_aug_gap,
                    result.avg_aug_gap,
                )
            )
        elif solved_instance_num > 0:
            self.logger.info(
                "Avg gap unavailable because public optimal tour lengths were not provided "
                "for the evaluated instances."
            )

        if detailed_log:
            self.logger.info("===============================================================")
            self.logger.info("instance: {}".format(result.instances))
            self.logger.info("optimal: {}".format(result.optimal))
            self.logger.info("problem_size: {}".format(result.problem_size))
            self.logger.info("no_aug_score: {}".format(result.no_aug_score))
            self.logger.info("aug_score: {}".format(result.aug_score))
            self.logger.info("no_aug_gap: {}".format(result.no_aug_gap))
            self.logger.info("aug_gap: {}".format(result.aug_gap))

        return result

    # def _test_one_instance(self, nodes_xy_normalized: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> Tuple[float, float]:
    #     if self.tester_params['augmentation_enable']:
    #         aug_factor = self.tester_params['aug_factor']
    #         if aug_factor != 8:
    #             raise NotImplementedError('Only aug_factor=8 is supported.')
    #     else:
    #         aug_factor = 1

    #     problems = nodes_xy_normalized
    #     if aug_factor > 1:
    #         problems = augment_xy_data_by_8_fold(problems)

    #     effective_batch = problems.size(0)
    #     problem_size = problems.size(1)

    #     env = Env(problem_size=problem_size, pomo_size=problem_size)

    #     env.batch_size = effective_batch
    #     env.problems = problems.to(self.device)
    #     env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(effective_batch, env.pomo_size)
    #     env.POMO_IDX = torch.arange(env.pomo_size, device=self.device)[None, :].expand(effective_batch, env.pomo_size)

    #     # Unify TSPLIB scoring: let Env compute integer tour length.
    #     # - original coords are used for TSPLIB cost (not normalized)
    #     # - edge_weight_type controls EUC_2D vs CEIL_2D discretization
    #     env.original_node_xy_lib = coords_orig[None, :, :]
    #     env.edge_weight_type = ew_type

    #     self.model.eval()
    #     with torch.no_grad():
    #         reset_state, _, _ = env.reset()
    #         self.model.pre_forward(reset_state)

    #         state, reward, done = env.pre_step()
    #         while not done:
    #             selected, _ = self.model(state)
    #             state, reward, done = env.step(selected, lib_mode=True)

    #     # reward is negative tour length at the final step
    #     tour_lengths = -reward
    #     best_len_per_aug = tour_lengths.min(dim=1).values
    #     no_aug_score = best_len_per_aug[0].item()
    #     aug_score = best_len_per_aug.min(dim=0).values.item()

    #     return float(no_aug_score), float(aug_score)

    ######################### 融入多重采样和2-opt的版本##########################
    def _test_one_instance(self, nodes_xy_normalized: torch.Tensor,
                           coords_orig: torch.Tensor, ew_type: str
                           ) -> Tuple[float, float]:
        # ------ 读取参数 ------
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            if aug_factor != 8:
                raise NotImplementedError('Only aug_factor=8 is supported.')
        else:
            aug_factor = 1

        num_samples = self.tester_params.get('num_samples', 1)
        enable_2opt = self.tester_params.get('enable_2opt', False)

        # ------ 数据增强 ------
        problems = nodes_xy_normalized
        if aug_factor > 1:
            problems = augment_xy_data_by_8_fold(problems)

        effective_batch = problems.size(0)   # 1 或 8（有增强时）
        problem_size = problems.size(1)

        # ------ 初始化环境 ------
        env = Env(problem_size=problem_size, pomo_size=problem_size)
        env.batch_size = effective_batch
        env.problems = problems.to(self.device)
        env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(
            effective_batch, env.pomo_size)
        env.POMO_IDX = torch.arange(env.pomo_size, device=self.device)[None, :].expand(
            effective_batch, env.pomo_size)
        env.original_node_xy_lib = coords_orig[None, :, :]
        env.edge_weight_type = ew_type

        # ------ 预计算距离矩阵（2-opt 用）------
        dist_matrix = None
        if enable_2opt:
            dist_matrix = _compute_dist_matrix(coords_orig, ew_type)

        # ------ 多采样推理 ------
        self.model.eval()
        all_sample_best = []     # 每次采样的最优路线长度

        with torch.no_grad():
            # Encoder 只算一次（所有采样共享同一个编码结果）
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            saved_eval = self.model.model_params['eval_type']

            for sample_id in range(num_samples):
                # 前 N-1 次用 softmax 随机采样，最后一次用 argmax 贪心
                self.model.model_params['eval_type'] = (
                    'softmax' if num_samples > 1 and sample_id < num_samples - 1
                    else 'argmax'
                )

                # 手动重置环境的动态状态（不需要重新编码，所以不调 env.reset()）
                env.selected_count = 0
                env.current_node = None
                env.selected_node_list = torch.zeros(
                    (effective_batch, env.pomo_size, 0),
                    dtype=torch.long, device=self.device
                )
                env.step_state = Step_State(
                    BATCH_IDX=env.BATCH_IDX, POMO_IDX=env.POMO_IDX
                )
                env.step_state.ninf_mask = torch.zeros(
                    (effective_batch, env.pomo_size, problem_size),
                    device=self.device
                )

                # Decoder 逐步选城市
                state, reward, done = env.pre_step()
                while not done:
                    selected, _ = self.model(state)
                    state, reward, done = env.step(selected, lib_mode=True)

                # reward 是负路程，取反得到实际路程
                tour_lengths = -reward   # shape: (effective_batch, pomo_size)

                # ------ 可选：2-opt 后处理 ------
                if enable_2opt:
                    batch_best_lengths = []
                    best_pomo_idx = tour_lengths.argmin(dim=1)
                    for b in range(effective_batch):
                        pidx = best_pomo_idx[b].item()
                        tour = env.selected_node_list[b, pidx].cpu().tolist()
                        optimized = _two_opt(tour, dist_matrix)
                        opt_len = _tour_length(optimized, dist_matrix)
                        batch_best_lengths.append(opt_len)
                    all_sample_best.append(
                        torch.tensor(batch_best_lengths, dtype=torch.float32)
                    )
                else:
                    # 不做 2-opt，直接取每个 aug batch 里最短的 POMO 路线
                    all_sample_best.append(tour_lengths.min(dim=1).values.cpu())

            # 恢复原来的 eval_type
            self.model.model_params['eval_type'] = saved_eval

        # ------ 合并所有采样结果 ------
        all_sample_best = torch.stack(all_sample_best, dim=0)
        # shape: (num_samples, effective_batch)

        # no_aug_score: 第 0 个 batch（无增强版本）在所有采样中的最优
        no_aug_score = all_sample_best[:, 0].min().item()
        # aug_score: 所有采样 x 所有增强版本中的全局最优
        aug_score = all_sample_best.min().item()

        return float(no_aug_score), float(aug_score)