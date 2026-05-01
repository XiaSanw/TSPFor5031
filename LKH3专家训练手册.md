# LKH3 + Behavior Cloning + RL Fine-tuning 操作手册

> **目标**：用 LKH3 生成 TSP 专家数据，通过 Behavior Cloning (BC) 预训练模型，再用 REINFORCE 做 RL Fine-tuning。核心贡献：利用 POMO 多起点结构，将一条 LKH3 tour 旋转为 N 条等价监督信号（N=problem_size），数据效率提升 N 倍。

---

## 一、原理速览

```
Baseline (纯 POMO)                 LKH3 + BC + RL (本方案)
     ↓                                     ↓
  随机初始化                        LKH3 生成最优 tour
     ↓                                     ↓
  REINFORCE 从零探索               BC：模仿 LKH3 每一步决策
     ↓                              (POMO 多轨迹 × 旋转 = N× 监督)
  慢、震荡、方差大                       ↓
     ↓                              RL Fine-tuning：超越老师
  gap ~2.33%                        (冻结 encoder 过渡，防崩塌)
                                         ↓
                                    gap 目标 <1.0%
```

**为什么有效**：
- LKH3 是当前最强 TSP 求解器之一，n≤200 几乎总能找到最优解
- BC 让模型快速学到"好的局部选择模式"（避免交叉边、优先近邻等）
- **POMO 的关键优势**：一条 LKH3 tour `[0→3→7→...]` 旋转后，可以同时监督全部 N 条 POMO trajectory（N=problem_size），数据利用率 N 倍于普通 BC
- RL Fine-tuning 在好的初始化基础上，探索比 LKH3 更优的策略
- **合规**：LKH3 只在训练阶段使用，测试时纯模型前向推理，不违反助教关于禁止 2-opt/LKH3 后处理的规定

---

## 二、环境准备：安装 LKH3

### 2.1 下载与编译

```bash
cd ~
wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.10.tgz
tar -xzf LKH-3.0.10.tgz
cd LKH-3.0.10
make

# 确认可执行文件存在
./LKH --help
```

> 如果服务器无外网，可以本地下载后 `scp` 上传。
> **注意**：若上述链接失效，可尝试：
> - LKH3 官方站点：http://akira.ruc.dk/~keld/research/LKH-3/
> - 或使用 LKH-2（n≤200 效果等同）：http://akira.ruc.dk/~keld/research/LKH/

### 2.2 配置环境变量

```bash
# 在 ~/.bashrc 或 ~/.zshrc 中添加
export LKH3_PATH="$HOME/LKH-3.0.10/LKH"
```

---

## 三、Step 1：生成专家数据

### 3.1 关键设计决策

| 问题 | 决定 | 原因 |
|------|------|------|
| 数据量 | n=100: 5万条; n=200: 5万条; n=150: 3万条 | 3 天时间约束，n=250 留给 RL 阶段不生成 BC 数据 |
| 数据格式 | `(problem, tour)` 对，`.pt` 文件 | 直接可用，无需预处理 |
| 坐标范围 | [0, 1] 浮点 → 放大到 [0, 1000] 写 TSPLIB | LKH3 接受浮点但放大避免精度问题 |
| 并行策略 | 多进程，每进程处理一批 | 10 万条串行太慢，需并行 |

### 3.2 数据生成脚本

创建 `TSP/POMO/generate_lkh3_data.py`：

```python
"""
生成随机 TSP instances，用 LKH3 求解，保存为 (problem, tour) 对。
支持多进程并行加速。
"""
import os
import sys
import subprocess
import torch
import numpy as np
from multiprocessing import Pool

# 添加父目录到 path，以便 import TSProblemDef
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

LKH3_BIN = os.environ.get("LKH3_PATH", "./LKH")
DATA_DIR = "./lkh3_expert_data"
NUM_WORKERS = 8  # 并行进程数


def problem_to_tsplib(problem, filename):
    """problem: (N, 2) numpy array, coordinates in [0, 1] → TSPLIB .tsp 文件"""
    n = problem.shape[0]
    scale = 1000.0
    with open(filename, 'w') as f:
        f.write(f"NAME: random_{n}\n")
        f.write(f"TYPE: TSP\n")
        f.write(f"DIMENSION: {n}\n")
        f.write(f"EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write(f"NODE_COORD_SECTION\n")
        for i in range(n):
            f.write(f"{i+1} {problem[i, 0]*scale:.6f} {problem[i, 1]*scale:.6f}\n")
        f.write("EOF\n")


def run_lkh3(tsp_file, tour_file, par_file):
    """调用 LKH3 求解 .tsp 文件"""
    with open(par_file, 'w') as f:
        f.write(f"PROBLEM_FILE = {tsp_file}\n")
        f.write(f"TOUR_FILE = {tour_file}\n")
        f.write(f"RUNS = 1\n")
        f.write(f"SEED = 0\n")

    result = subprocess.run(
        [LKH3_BIN, par_file],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"LKH3 failed: {result.stderr}")
    return tour_file


def parse_tour_file(tour_file, n):
    """解析 LKH3 .tour 文件，返回 0-based 城市索引列表"""
    tour = []
    in_tour = False
    with open(tour_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "TOUR_SECTION":
                in_tour = True
                continue
            if line == "-1" or line == "EOF":
                break
            if in_tour and line:
                tour.append(int(line) - 1)  # 1-based → 0-based
    assert len(tour) == n, f"Tour length mismatch: {len(tour)} != {n}"
    return tour


def solve_single_instance(args):
    """单个 instance 的完整求解流程（供多进程调用）"""
    i, problem, problem_size, output_dir = args
    tsp_file = os.path.join(output_dir, f"prob_{i}.tsp")
    tour_file = os.path.join(output_dir, f"prob_{i}.tour")
    par_file = os.path.join(output_dir, f"prob_{i}.par")

    problem_to_tsplib(problem, tsp_file)
    run_lkh3(tsp_file, tour_file, par_file)
    tour = parse_tour_file(tour_file, problem_size)

    # 清理临时文件
    os.remove(tsp_file)
    os.remove(tour_file)
    os.remove(par_file)

    return {'problem': problem, 'tour': np.array(tour, dtype=np.int64)}


def generate_dataset(num_instances, problem_size, output_dir, num_workers=NUM_WORKERS):
    """并行生成一批数据并求解"""
    os.makedirs(output_dir, exist_ok=True)
    from TSProblemDef import get_random_problems

    problems = get_random_problems(num_instances, problem_size).numpy()
    # shape: (N, problem_size, 2)

    # 准备多进程参数
    args_list = [
        (i, problems[i], problem_size, output_dir)
        for i in range(num_instances)
    ]

    # 多进程求解
    all_data = []
    with Pool(num_workers) as pool:
        for result in pool.imap_unordered(solve_single_instance, args_list):
            all_data.append(result)
            if len(all_data) % 1000 == 0:
                print(f"  Progress: {len(all_data)}/{num_instances}")

    # 保存为 pt 文件
    save_path = os.path.join(output_dir, f"lkh3_data_n{problem_size}.pt")
    torch.save(all_data, save_path)
    print(f"Saved {num_instances} instances to {save_path}")


if __name__ == "__main__":
    # 建议：按照你的时间/算力调整数据量
    # n=100 最快 (~2h/5万条 8核)，n=200 较慢 (~6h/5万条 8核)
    generate_dataset(num_instances=50000, problem_size=100,
                     output_dir=f"{DATA_DIR}/n100")
    generate_dataset(num_instances=50000, problem_size=200,
                     output_dir=f"{DATA_DIR}/n200")
    generate_dataset(num_instances=30000, problem_size=150,
                     output_dir=f"{DATA_DIR}/n150")
```

### 3.3 执行生成

```bash
cd TSP/POMO
export LKH3_PATH="$HOME/LKH-3.0.10/LKH"

# 建议用 nohup 或 screen（10 万+ 条 instance 可能需要数小时）
nohup python generate_lkh3_data.py > generate.log 2>&1 &

# 监控进度
tail -f generate.log
```

**预期时间**（8 核并行）：

| 规模 | 数据量 | 预计时间 |
|------|--------|----------|
| n=100 | 5 万条 | ~2 小时 |
| n=150 | 3 万条 | ~2 小时 |
| n=200 | 5 万条 | ~6 小时 |

**总计约 10 小时**，在 3 天时间预算内完全可行。

---

## 四、Step 2：Behavior Cloning 预训练

### 4.1 核心思路：POMO 多轨迹 × Tour 旋转 = N 倍监督

LKH3 给你一条最优 tour：`[0, 3, 7, 1, 5]`

**普通 BC（只监督 1 条轨迹）：**

```
POMO traj 0 (从城市 0 开始): expert = [3, 7, 1, 5]  <- 只用这一条
POMO traj 1 (从城市 3 开始): expert = ???           <- 浪费了
```

**高效 BC（本文方案，监督全部 N 条轨迹）：**

```
LKH3 tour: [0 -> 3 -> 7 -> 1 -> 5 -> 0]

POMO traj 0 (从城市 0 开始): expert = [3, 7, 1, 5]  <- tour 从 0 的位置顺推
POMO traj 1 (从城市 3 开始): expert = [7, 1, 5, 0]  <- tour 旋转到以 3 开头
POMO traj 2 (从城市 7 开始): expert = [1, 5, 0, 3]  <- tour 旋转到以 7 开头
...
```

**1 条 LKH3 数据 -> N 条有效监督信号（N=problem_size），n=100 时数据效率提升 100 倍。**

### 4.2 完整 BC 实现

在 `TSPTrainer.py` 的 `TSPTrainer` 类中添加以下方法（放在 `_train_one_batch` 方法之后）：

```python
def _train_one_epoch_bc(self, epoch, bc_data_loader):
    """
    Behavior Cloning 预训练一个 epoch。

    关键设计：
    1. 将 LKH3 tour 旋转，为每个 POMO 起点构造对应的 expert action 序列
    2. Teacher forcing：env 按 expert action 走，模型每一步接受监督
    3. Loss 覆盖全部 N 条 trajectory（N=problem_size），而非仅第 0 条
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

        # ---- 1. 注入 problems 到 env ----
        self.env.problem_size = problem_size
        self.env.pomo_size = pomo_size
        self.env.batch_size = batch_size
        self.env.problems = problems
        self.env.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, pomo_size)
        self.env.POMO_IDX = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)

        # ---- 2. 构造 expert action 矩阵（vectorized）----
        # tour_positions[b, city] = city 在 tours[b] 中的索引
        tour_positions = torch.zeros(batch_size, problem_size, dtype=torch.long, device=device)
        positions = torch.arange(problem_size, device=device).unsqueeze(0).expand(batch_size, -1)
        tour_positions.scatter_(1, tours, positions)

        # 将 tours 首尾拼接，避免取模运算
        cyclic_tours = torch.cat([tours, tours], dim=1)  # (B, 2N)

        # start_positions: (B, N) — 每个起点在 tour 中的位置
        # offsets: (N-1,) — [1, 2, ..., N-1]
        # indices: (B, N*(N-1)) — 展平后 gather，再 reshape 回 (B, N, N-1)
        offsets = torch.arange(1, problem_size, device=device)                # (N-1,)
        indices_3d = tour_positions[:, :, None] + offsets[None, None, :]      # (B, N, N-1)
        indices_flat = indices_3d.reshape(batch_size, -1)                     # (B, N*(N-1))

        expert_actions = torch.gather(cyclic_tours, 1, indices_flat)          # (B, N*(N-1))
        expert_actions = expert_actions.reshape(batch_size, pomo_size, problem_size - 1)
        # expert_actions[b, s, t] = 轨迹 s 在第 t+1 步的专家动作

        # ---- 3. Env reset + Encoder forward ----
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        # ---- 4. POMO rollout with teacher forcing ----
        prob_list = []  # 收集每步的 (B, POMO) 概率
        state, reward, done = self.env.pre_step()

        for step in range(problem_size):
            if step == 0:
                # Step 0: POMO 初始化（选起点）
                # 每条 POMO 轨迹以不同城市开始：轨迹 i 从城市 i 出发
                selected = torch.arange(pomo_size, device=device)[None, :].expand(batch_size, pomo_size)

                # 设置 decoder 的 q_first（首节点 embedding），与 TSPModel.forward 一致
                encoded_first_node = _get_encoding(self.model.encoded_nodes, selected)
                self.model.decoder.set_q1(encoded_first_node)

                state, reward, done = self.env.step(selected)
            else:
                # 获取模型对当前状态的 action 概率分布
                encoded_last_node = _get_encoding(self.model.encoded_nodes, state.current_node)
                probs = self.model.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
                # probs: (B, POMO, N)

                # 取专家动作对应的概率（clamp 防止 log(0) = -inf）
                expert = expert_actions[:, :, step - 1]  # (B, POMO)
                prob = probs[self.env.BATCH_IDX, self.env.POMO_IDX, expert]  # (B, POMO)
                prob_list.append(prob)

                # Teacher forcing：环境按专家动作走
                state, reward, done = self.env.step(expert)

        # ---- 5. BC Loss ----
        # prob_list 有 N-1 个元素，每个 (B, POMO)
        all_probs = torch.stack(prob_list, dim=2)                # (B, POMO, N-1)
        log_probs = torch.clamp(all_probs, min=1e-8).log()      # 数值稳定
        total_log_prob = log_probs.sum(dim=2)                    # (B, POMO)
        loss = -total_log_prob.mean()                            # scalar

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        loss_AM.update(loss.item(), batch_size)

    self.logger.info('Epoch {:3d}: BC Loss: {:.4f}'.format(epoch, loss_AM.avg))
    return loss_AM.avg
```

### 4.3 BC 数据加载器

创建 `TSP/POMO/bc_dataset.py`：

```python
import os
import re
import torch
from torch.utils.data import Dataset, DataLoader


class TSPBCDataset(Dataset):
    """加载 LKH3 生成的 (problem, tour) 数据"""

    def __init__(self, data_path):
        self.data = torch.load(data_path, map_location='cpu')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'problem': torch.from_numpy(item['problem']).float(),
            'tour': torch.from_numpy(item['tour']).long()
        }


def get_bc_dataloader(data_path, batch_size, shuffle=True, num_workers=2):
    """
    单尺度 BC 数据加载器。

    num_workers 建议：
    - 2：安全，大多数服务器没问题
    - 0：数据集较小时（<5 万条/规模）或内存紧张时
    - 4+：数据集非常大时可用，但注意每个 worker 会 fork 一份原始数据的引用，
           内存紧张时可能导致 OOM
    """
    dataset = TSPBCDataset(data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def build_bc_loaders(data_paths, batch_size, num_workers=2):
    """
    为每个规模构建独立的 DataLoader。

    重要：不同规模的样本不能混在同一个 batch 中（tensor shape 不同，collate 会报错）。
    训练时每个 epoch 随机选一个规模进行 BC，既保证 mixed-size 又避免 shape mismatch。
    """
    loaders = {}
    for path in data_paths:
        # 从文件名推断规模，如 lkh3_data_n100.pt → 100
        match = re.search(r'n(\d+)', os.path.basename(path))
        size = int(match.group(1)) if match else 0
        loaders[size] = get_bc_dataloader(path, batch_size, num_workers=num_workers)

    return loaders
```

### 4.4 修改 train.py：添加 BC 阶段 + RL 过渡

```python
# ========== 新增：BC 预训练配置 ==========
BC_CONFIG = {
    'enable': True,
    'data_paths': [
        './lkh3_expert_data/n100/lkh3_data_n100.pt',
        './lkh3_expert_data/n150/lkh3_data_n150.pt',
        './lkh3_expert_data/n200/lkh3_data_n200.pt',
    ],
    'epochs': 200,           # BC 训练 epoch 数
    'batch_size': 64,        # BC batch size（teacher forcing 内存省，可设大）
    'lr': 1e-4,              # BC 学习率（比 RL 小，防止遗忘）
    # BC → RL 过渡策略
    'freeze_encoder_epochs': 100,   # RL 前 100 epoch 冻结 encoder
    'rl_warmup_epochs': 50,         # RL 前 50 epoch 学习率线性 warmup
    'rl_warmup_lr_start': 1e-5,     # warmup 起始学习率
}
```

修改 `main()` 函数：

```python
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
    # Phase 1: Behavior Cloning 预训练
    # =================================================================
    if BC_CONFIG['enable']:
        from bc_dataset import build_bc_loaders

        logger = logging.getLogger('root')
        logger.info("=" * 60)
        logger.info("Phase 1: Behavior Cloning Pretraining")
        logger.info("=" * 60)

        # 每个规模独立的 DataLoader，避免不同 shape 的 tensor 混在一个 batch
        bc_loaders = build_bc_loaders(
            BC_CONFIG['data_paths'],
            batch_size=BC_CONFIG['batch_size'],
            num_workers=2
        )

        # BC 阶段用独立的 optimizer，避免污染 RL 阶段的 Adam 动量
        bc_optimizer = torch.optim.Adam(
            trainer.model.parameters(),
            lr=BC_CONFIG['lr'],
            weight_decay=optimizer_params['optimizer'].get('weight_decay', 1e-6)
        )
        # 临时替换 trainer 的 optimizer
        original_optimizer = trainer.optimizer
        trainer.optimizer = bc_optimizer

        # BC 训练循环：每个 epoch 随机选一个规模
        import random as _random
        for bc_epoch in range(1, BC_CONFIG['epochs'] + 1):
            size = _random.choice(list(bc_loaders.keys()))
            bc_loss = trainer._train_one_epoch_bc(bc_epoch, bc_loaders[size])
            # 每 50 epoch 打印一次
            if bc_epoch % 50 == 0 or bc_epoch == 1:
                logger.info('BC Epoch {:3d}/{:3d} (n={}): Loss = {:.4f}'.format(
                    bc_epoch, BC_CONFIG['epochs'], size, bc_loss))

        # 保存 BC 预训练权重
        bc_ckpt_path = '{}/bc_pretrained.pt'.format(trainer.result_folder)
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
        }, bc_ckpt_path)
        logger.info('BC Pretraining Done! Saved to {}'.format(bc_ckpt_path))

        # =============================================================
        # BC → RL 过渡准备
        # =============================================================
        # 1. 冻结 encoder（防止 RL 初期策略漂移破坏 BC 学到的表示）
        if BC_CONFIG['freeze_encoder_epochs'] > 0:
            for param in trainer.model.encoder.parameters():
                param.requires_grad = False
            logger.info('Encoder frozen for first {} RL epochs'.format(
                BC_CONFIG['freeze_encoder_epochs']))

        # 2. 为 RL 阶段重建 optimizer + scheduler
        #    base_lr 设为目标 lr (3e-4)，由 LinearLR 的 start_factor 控制 warmup 起点
        #    这样 epoch 0: lr = 3e-4 * (1e-5/3e-4) = 1e-5，warmup 结束: lr = 3e-4
        rl_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, trainer.model.parameters()),
            lr=optimizer_params['optimizer']['lr'],       # base_lr = 3e-4（目标 lr）
            weight_decay=optimizer_params['optimizer'].get('weight_decay', 1e-6)
        )
        trainer.optimizer = rl_optimizer
        logger.info('RL optimizer rebuilt, base_lr = {}, warmup from {}'.format(
            optimizer_params['optimizer']['lr'], BC_CONFIG['rl_warmup_lr_start']))

        # 3. 重建 scheduler（覆盖原来的 warmup+cosine，避免冲突）
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

        # 将过渡参数注入 trainer（用于 encoder 解冻）
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
```

---

## 五、Step 3：RL Fine-tuning 过渡机制

### 5.1 BC → RL 过渡策略

这是整个 pipeline 最关键的工程决策。直接从 BC 切 RL 容易**策略崩塌（policy collapse）**——REINFORCE 探索到 BC 没见过的状态，给出错误动作，进入恶性循环。

三层保护机制：

```
Layer 1: 冻结 Encoder（前 100 epoch 只训 Decoder）
         ↓  BC 学到的城市表示不会被动摇
Layer 2: 学习率 Warmup（由重建的 SequentialLR 管理）
         ↓  避免 RL 初期大步更新破坏策略
Layer 3: Leader Reward（已启用），POMO baseline
         ↓  降低 REINFORCE 方差
```

### 5.2 修改 TSPTrainer.run()：支持 Encoder 解冻

在 `TSPTrainer.__init__` 末尾添加：

```python
# === BC → RL transition config ===
self.bc_transition = None  # 由 train.py 注入
```

修改 `TSPTrainer.run()` 的 epoch 循环（只需处理 encoder 解冻，LR 由 scheduler 管理）：

```python
def run(self):
    self.time_estimator.reset(self.start_epoch)
    for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
        self.logger.info('=================================================================')

        # === BC → RL 过渡：解冻 encoder ===
        if (self.bc_transition is not None
                and self.bc_transition.get('freeze_encoder_epochs', 0) > 0
                and epoch == self.bc_transition['freeze_encoder_epochs'] + 1):
            for param in self.model.encoder.parameters():
                param.requires_grad = True
            # 解冻后需要将 encoder 参数加回 optimizer
            self.optimizer.add_param_group({
                'params': list(self.model.encoder.parameters())
            })
            self.logger.info('>>> Encoder UNFROZEN at epoch {}, params added to optimizer <<<'.format(epoch))

        # Train
        train_score, train_loss = self._train_one_epoch(epoch)
        self.scheduler.step()
        # ... 后续 logging / checkpoint 不变 ...
```

完整修改见附录 A。

---

## 六、完整执行流程（3 天 CheckList）

```bash
# ═════════ Day 1 上午：环境准备 ═════════
# 1. 安装 LKH3
cd ~ && wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.10.tgz
tar -xzf LKH-3.0.10.tgz && cd LKH-3.0.10 && make
export LKH3_PATH="$HOME/LKH-3.0.10/LKH"

# 2. 生成专家数据（后台运行，利用午休时间）
cd TSP/POMO
nohup python generate_lkh3_data.py > generate.log 2>&1 &

# ═════════ Day 1 下午 ~ 晚上：代码修改 ═════════
# 3. 按本手册修改代码：
#    - TSPTrainer.py: 添加 _train_one_epoch_bc + bc_transition 支持
#    - 新建 bc_dataset.py
#    - train.py: 添加 BC_CONFIG + 修改 main()
# 数据生成完成后检查:
# ls -lh lkh3_expert_data/n100/lkh3_data_n100.pt

# ═════════ Day 1 晚上 ~ Day 2 上午：BC 预训练 ═════════
# 4. 启动 BC 训练
python train.py
# 预期：200 epoch BC，每 epoch ~2-5 分钟（取决于 batch size 和 GPU）

# ═════════ Day 2 上午 ~ Day 3 下午：RL Fine-tuning ═════════
# BC 结束后自动进入 RL 阶段
# 前 100 epoch: encoder frozen + lr warmup
# 100 epoch 后: encoder 解冻，余弦退火学习率
# 预期总共 3000-5000 epoch，A100 约 24-36 小时

# ═════════ Day 3 下午~晚上：测试 ═════════
# 5. 测试（合规版本：无 2-opt，aug=8）
python test.py \
  --checkpoint_path <result_dir>/checkpoint-<best_epoch>.pt \
  --augmentation_enable true \
  --aug_factor 8
```

---

## 七、预期效果与调参建议

### 7.1 预期效果

| 阶段 | 预期 gap（n=100 TSPLIB） | 说明 |
|------|------------------------|------|
| Baseline（纯 POMO） | ~2.33% | 随机初始化 REINFORCE |
| BC 预训练后（不 RL） | ~1.0-1.5% | 纯模仿 LKH3，上限被锁 |
| **BC + RL（本方案）** | **目标 <1.0%** | 不依赖后处理，纯模型输出 |
| 参考：多采样+2opt（已禁） | ~0.67% | 助教禁止，仅作参考上限 |

### 7.2 调参建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| BC 数据量 | 5-10 万条/size | 底线 5 万，更多更好 |
| BC epoch | 150-300 | loss 还在降就多跑，看 validation gap |
| BC lr | 1e-4 | 比 RL 小，BC 收敛更平滑 |
| Encoder freeze epoch | 50-150 | 太长->创新空间被锁死；太短->容易崩塌 |
| RL warmup epoch | 30-80 | 配合 freeze epoch |
| RL warmup 起始 lr | 5e-6 ~ 2e-5 | 越小越稳，越慢 |

### 7.3 如何选 Best Epoch

BC 阶段无法直接测 validation gap（BC 训的是 log_prob，不是 tour length）。建议：

```bash
# 每 50 BC epoch 保存一次，用 test.py 跑 validation
python test.py --checkpoint_path <dir>/bc_epoch_50.pt --augmentation_enable true

# 选 validation gap 最低的那个 checkpoint 进入 RL
```

---

## 八、常见问题

**Q: BC 后 RL 阶段 loss 暴增 / score 骤降怎么办？**

A: Policy collapse。按优先级尝试：
1. 增大 `freeze_encoder_epochs`（如 100 -> 200）
2. 降低 `rl_warmup_lr_start`（如 1e-5 -> 5e-6）
3. 延长 `rl_warmup_epochs`（如 50 -> 100）
4. 混合训练：每个 RL epoch 先跑 1 batch BC 数据再跑 RL

**Q: LKH3 数据太少/太多？**

A: 时间紧就各 3 万条 + 多跑 BC epoch（250-300）。时间充裕就各 10 万条 + 少跑 BC epoch（150-200）。

**Q: BC 阶段需不需要 mixed_sizes？**

A: 用。BC data loader 本身就混合了 n=100/150/200，和 RL 阶段的 mixed_sizes 衔接自然。n=250 留给 RL 阶段，不生成 BC 数据。

**Q: 可以只用 BC 不跑 RL 吗？**

A: 可以，但 gap 天花板就是 LKH3 水平（~1%）。RL 是超越 LKH3 的唯一途径。

**Q: 助教禁止 2-opt 后处理，这个方案合规吗？**

A: 完全合规。LKH3 只在**训练阶段**当 teacher 用，测试时纯模型推理（forward pass），不调用任何启发式求解器。

---

## 附录 A：TSPTrainer 完整修改清单

### A.1 `__init__` 末尾追加

```python
# === BC → RL transition config ===
self.bc_transition = None  # 由 train.py 注入
```

### A.2 `run()` 方法 epoch 循环开头插入

在 `run()` 方法的 `for epoch in range(...):` 下一行，`self.logger.info('=====')` 之前插入：

```python
# === BC → RL 过渡：解冻 encoder ===
if (self.bc_transition is not None
        and self.bc_transition.get('freeze_encoder_epochs', 0) > 0
        and epoch == self.bc_transition['freeze_encoder_epochs'] + 1):
    for param in self.model.encoder.parameters():
        param.requires_grad = True
    # 解冻后需要将 encoder 参数加回 optimizer
    self.optimizer.add_param_group({
        'params': list(self.model.encoder.parameters())
    })
    self.logger.info('>>> Encoder UNFROZEN at epoch {}, params added to optimizer <<<'.format(epoch))
```

### A.3 插入 `_train_one_epoch_bc` 方法

见第四章 4.2 节完整代码。

---

## 附录 B：项目文件结构总览

```
TSP/POMO/
├── train.py                   # <- 修改：添加 BC_CONFIG + Phase 1/2 逻辑
├── TSPTrainer.py              # <- 修改：添加 _train_one_epoch_bc + bc_transition
├── TSPModel.py                # 不变
├── TSPEnv.py                  # 不变
├── test.py                    # 不变（测试入口）
├── TSPTester_LIB.py           # 不变
├── tsplib_utils.py            # 不变
├── bc_dataset.py              # <- 新建
├── generate_lkh3_data.py      # <- 新建
└── lkh3_expert_data/          # <- LKH3 生成的数据目录
    ├── n100/
    │   └── lkh3_data_n100.pt
    ├── n150/
    │   └── lkh3_data_n150.pt
    └── n200/
        └── lkh3_data_n200.pt
```

---

## 附录 C：修订记录

以下是本次修订相对于原版手册的所有改动：

### C.1 严重 Bug 修复

| # | 位置 | 原始问题 | 修复内容 |
|---|------|---------|---------|
| 1 | 4.2 `_train_one_epoch_bc` expert_actions 构造 | `torch.gather(cyclic_tours, 1, indices)` 维度不匹配：`cyclic_tours` 是 2D `(B,2N)`，`indices` 是 3D `(B,N,N-1)`，运行时直接 RuntimeError | 先将 `indices` reshape 为 `(B, N*(N-1))` 做 gather，再 reshape 回 `(B, N, N-1)` |
| 2 | 4.2 `tour_positions` 构造 | 用 Python 双重 for 循环 `for b / for pos, city`，batch=64 n=200 时 12800 次迭代，极慢 | 改为 `scatter_` 向量化：`tour_positions.scatter_(1, tours, positions)` |
| 3 | 4.2 step==0 | 缺少 `decoder.set_q1()` 调用。`TSPModel.forward()` 在 `current_node is None` 时会调 `set_q1(encoded_first_node)` 初始化首节点 query，手册跳过导致后续 decoder 输出错误 | 在 step==0 分支添加 `_get_encoding` + `self.model.decoder.set_q1(encoded_first_node)` |
| 4 | 4.4 + 5.2 LR scheduler | 手册在 `run()` 开头手动设 LR，但紧接着 `self.scheduler.step()` (cosine annealing) 会立刻覆盖手动值，导致 BC->RL warmup 完全失效 | 在 `main()` 中 BC 结束后重建 RL optimizer + SequentialLR(warmup+cosine)，`run()` 中只保留 `scheduler.step()`，不再手动设 LR |
| 4b | 4.4 rl_optimizer base_lr | rl_optimizer 的 `lr` 设为 `rl_warmup_lr_start` (1e-5)，再叠加 `LinearLR(start_factor=1e-5/3e-4=0.033)` → epoch 0 实际 lr = 1e-5 * 0.033 = 3.3e-7，远低于预期 | base_lr 改为目标 lr (3e-4)，`LinearLR(start_factor=1e-5/3e-4)` → epoch 0: lr = 3e-4 * 0.033 = 1e-5 ✅ |

### C.2 概念错误修正

| # | 位置 | 原始问题 | 修复内容 |
|---|------|---------|---------|
| 5 | 全文多处 | "数据效率提升 8 倍"、"8 条 POMO trajectory" — 8 是 POMO 几何增强 (`augment_xy_data_by_8_fold`) 的倍数，与 tour 旋转无关。实际 `pomo_size = problem_size`，n=100 时是 100 倍 | 全部改为 "N 倍"（N=problem_size），并在 4.1 节明确举例 "n=100 时数据效率提升 100 倍" |

### C.3 代码健壮性改进

| # | 位置 | 原始问题 | 修复内容 |
|---|------|---------|---------|
| 6 | 4.2 `_train_one_epoch_bc` | 方法开头缺少 `self.model.train()`，若之前处于 eval 模式，BatchNorm/Dropout 行为不正确 | 添加 `self.model.train()` |
| 7 | 4.2 BC Loss | `torch.stack(prob_list).log()` 概率接近 0 时产生 -inf/NaN | 改为 `torch.clamp(all_probs, min=1e-8).log()` |
| 8 | 4.4 train.py | BC 和 RL 共用同一个 Adam optimizer，BC 阶段的动量历史在 RL 阶段会造成干扰；冻结 encoder 后 optimizer 仍包含其参数浪费内存 | BC 阶段用独立 `bc_optimizer`；RL 阶段重建 optimizer（只含 `requires_grad=True` 的参数） |
| 9 | 5.2 encoder 解冻 | 原版只设 `requires_grad = True`，但 encoder 参数不在 RL optimizer 的 param_groups 中，解冻后仍不会被更新 | 解冻时调用 `self.optimizer.add_param_group()` 将 encoder 参数加入 optimizer |
| 10 | 4.3 bc_dataset.py | `build_bc_loaders` 内部重复 `import re` | 移到文件顶部（已有） |
