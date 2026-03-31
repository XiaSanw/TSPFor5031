import os
import time
from dataclasses import dataclass
from logging import getLogger
from typing import List, Tuple

import numpy as np
import torch

from TSPEnv import TSPEnv as Env
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


@dataclass
class LibResult:
    instances: List[str]
    optimal: List[float]
    problem_size: List[int]
    no_aug_score: List[float]
    aug_score: List[float]
    no_aug_gap: List[float]
    aug_gap: List[float]


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
                        self.logger.info(f"Skip (optimal not found): {name}")
                        continue

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

                    no_aug_gap = (no_aug_score - optimal) / optimal * 100
                    aug_gap = (aug_score - optimal) / optimal * 100

                    result.instances.append(name)
                    result.optimal.append(float(optimal))
                    result.problem_size.append(int(dimension))
                    result.no_aug_score.append(float(no_aug_score))
                    result.aug_score.append(float(aug_score))
                    result.no_aug_gap.append(float(no_aug_gap))
                    result.aug_gap.append(float(aug_gap))

                    self.logger.info(
                        "optimal: {:.3f}, no_aug: {:.3f} (gap {:.3f}%), aug: {:.3f} (gap {:.3f}%)".format(
                            optimal, no_aug_score, no_aug_gap, aug_score, aug_gap
                        )
                    )

        end_time_all = time.time()
        self.logger.info("=========================== Summary ===========================")
        self.logger.info(
            "All done, solved instance number: {}/{}, total time: {:.2f}s".format(
                solved_instance_num, all_instance_num, end_time_all - start_time_all
            )
        )

        if solved_instance_num > 0:
            self.logger.info(
                "Avg gap(no aug): {:.3f}%, Avg gap(aug): {:.3f}%".format(
                    float(np.mean(result.no_aug_gap)),
                    float(np.mean(result.aug_gap)),
                )
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

    def _test_one_instance(self, nodes_xy_normalized: torch.Tensor, coords_orig: torch.Tensor, ew_type: str) -> Tuple[float, float]:
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            if aug_factor != 8:
                raise NotImplementedError('Only aug_factor=8 is supported.')
        else:
            aug_factor = 1

        problems = nodes_xy_normalized
        if aug_factor > 1:
            problems = augment_xy_data_by_8_fold(problems)

        effective_batch = problems.size(0)
        problem_size = problems.size(1)

        env = Env(problem_size=problem_size, pomo_size=problem_size)

        env.batch_size = effective_batch
        env.problems = problems.to(self.device)
        env.BATCH_IDX = torch.arange(effective_batch, device=self.device)[:, None].expand(effective_batch, env.pomo_size)
        env.POMO_IDX = torch.arange(env.pomo_size, device=self.device)[None, :].expand(effective_batch, env.pomo_size)

        # Unify TSPLIB scoring: let Env compute integer tour length.
        # - original coords are used for TSPLIB cost (not normalized)
        # - edge_weight_type controls EUC_2D vs CEIL_2D discretization
        env.original_node_xy_lib = coords_orig[None, :, :]
        env.edge_weight_type = ew_type

        self.model.eval()
        with torch.no_grad():
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected, lib_mode=True)

        # reward is negative tour length at the final step
        tour_lengths = -reward
        best_len_per_aug = tour_lengths.min(dim=1).values
        no_aug_score = best_len_per_aug[0].item()
        aug_score = best_len_per_aug.min(dim=0).values.item()

        return float(no_aug_score), float(aug_score)
