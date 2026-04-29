# 🧠 SDM-5031 TSP 项目代码完全指南

> 写给代码小白的零基础教程。读完这篇，你会知道每一行代码在干什么。

---

## 📁 项目文件结构（鸟瞰图）

```
SDM-5031-2026-Spring/
├── README.md                    ← 项目说明书（老师给的）
├── requirements.txt              ← 需要安装的 Python 包列表
├── 实验记录.md                  ← 你的实验数据记录
├── 改进方案.md                  ← 改进方案的详细设计
├── TSP/                          ← 🌟 核心代码都在这里
│   ├── TSProblemDef.py           ← 生成训练用的随机 TSP 问题
│   ├── data/
│   │   └── val/                  ← 公开验证集（用来调试）
│   └── POMO/                     ← 🌟 模型 + 训练 + 测试
│       ├── train.py              ← 🚀 训练入口：启动训练
│       ├── test.py               ← 🧪 测试入口：评估模型
│       ├── TSPModel.py           ← 🤖 神经网络模型定义（大脑）
│       ├── TSPEnv.py             ← 🌍 环境：模拟 TSP 走法
│       ├── TSPTrainer.py         ← 🏋️ 训练器：控制训练流程
│       ├── TSPTester_LIB.py      ← 📊 测试器：评估 TSPLIB 实例
│       ├── tsplib_utils.py       ← 📖 TSPLIB 文件读取工具
│       └── result/               ← 保存的训练结果和模型
│           └── saved_tsp100_model2_longTrain/
│               └── checkpoint-3000.pt   ← 🏆 Baseline 模型
└── utils/
    └── utils.py                  ← 日志、计时等通用工具
```

---

## 🎭 一句话概括每个文件

| 文件 | 作用 | 比喻 |
|------|------|------|
| `TSPModel.py` | 定义神经网络结构 | 🧠 **大脑**：决定下一步走哪个城市 |
| `TSPEnv.py` | 模拟 TSP 走路线过程 | 🌍 **地图**：记录已经去了哪些城市，计算路线长度 |
| `TSPTrainer.py` | 控制训练循环 | 🏋️ **教练**：告诉大脑"这条路好不好"，然后调整大脑 |
| `TSPTester_LIB.py` | 用真实数据集测试 | 📊 **考官**：拿 TSPLIB 真实题目考大脑 |
| `train.py` | 启动训练的配置文件 | 🚀 **发射按钮**：设置好参数，点击发射 |
| `test.py` | 启动测试的配置文件 | 🧪 **答题卡**：设置好试卷，让大脑做题 |
| `TSProblemDef.py` | 生成随机训练题目 | 🎲 **出题机**：自动生成练习用的 TSP 题目 |

---

## 🔄 训练流程全景图（数据怎么流动）

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  出题机     │ --> │  环境       │ --> │  大脑       │ --> │  教练       │
│  TSProblemDef│     │  TSPEnv     │     │  TSPModel   │     │  TSPTrainer │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │                    │
      ▼                    ▼                    ▼                    ▼
生成随机城市坐标      记录已走的城市       预测下一个城市       计算Loss，
                    计算路线长度         输出概率分布         反向传播更新参数
```

**一句话**：出题机生成题目 → 环境模拟走路线 → 大脑预测下一步 → 教练评价并改进大脑。

---

## 🔑 核心概念科普（必看！）

### 1. Batch（批次）

**是什么？**
> 一次训练同时处理多少个 TSP 实例。

**打个比方**：
> 你在做数学题练习。`batch_size = 64` 就是一次做 64 道题，算完 64 道题的平均成绩后再统一改错。

**为什么不用 1？**
> 做 1 道题就改错，改的方向可能很偏（这道题刚好很难）。做 64 道取平均，改的方向更靠谱。

**代码里的体现**：
```python
# TSPTrainer.py 中
train_batch_size = 64   # 一次处理64个TSP实例
```

**收益**：
- ✅ batch 越大，梯度越稳定，训练越靠谱
- ❌ batch 太大，显存（GPU内存）不够
- 💡 你的代码里 250 城市时 batch 会缩到 16，就是为了省显存

---

### 2. Learning Rate（学习率）

**是什么？**
> 每次改错时，调整的幅度有多大。

**打个比方**：
> 你在学投篮。学习率 = 0.1 就是"这次偏右了，下次往左调 10 厘米"。学习率 = 0.001 就是"这次偏右了，下次往左调 1 毫米"。

**太大了会怎样？**
> 像喝醉了调投篮——永远调过头，永远不准。

**太小了会怎样？**
> 像在月球上调投篮——每次只动一点点，一万年也调不到准星。

**代码里的体现**：
```python
# train.py 中
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,        # 学习率 = 0.0001
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [3401,],  # 第3401轮时降低学习率
        'gamma': 0.1              # 降到原来的 1/10
    }
}
```

**收益**：
- ✅ 初期学习率大 → 快速接近好答案
- ✅ 后期学习率小 → 精细微调，不走过头

---

### 3. Epoch（轮次）

**是什么？**
> 完整地把训练数据过一遍，叫 1 个 epoch。

**打个比方**：
> 你有一本题库（10万道题）。epoch = 1 就是把这 10 万道题全部做一遍。epoch = 3000 就是做了 3000 遍。

**代码里的体现**：
```python
# train.py 中
epochs = 3800   # 训练3800轮
```

**注意**：
> 你的 baseline 是训练了 3000 轮后的模型。你现在从 3000 续训到 3800，就是"再做 800 遍新题目"。

---

### 4. 采样（Sampling）

**是什么？**
> 大脑预测"下一步去城市 A 的概率是 70%，城市 B 是 20%，城市 C 是 10%"，采样就是按这个概率**随机选一个**。

**两种模式**：
- **Argmax（贪心）**：永远选概率最大的 → 同一条路线
- **Softmax 采样**：按概率随机选 → 不同次可能走不同路线

**打个比方**：
> 去餐厅吃饭。argmax = 永远去评分最高的那家。softmax采样 = 按评分概率随机选（评分高的更可能被选到，但偶尔也去别的）。

**代码里的体现**：
```python
# TSPModel.py 中 forward()
if self.training or self.model_params['eval_type'] == 'softmax':
    # 训练时：随机采样
    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1)
else:
    # 测试时：贪心选最大
    selected = probs.argmax(dim=2)
```

**收益**：
- 训练时采样 → 探索更多可能性，避免死脑筋
- 测试时贪心 → 选最稳的路线

---

### 5. 数据增强（Augmentation）

**是什么？**
> 同一组城市坐标，通过旋转、翻转、镜像等方式，生成 8 个"看起来不同但本质一样"的版本。

**打个比方**：
> 你拍了一张猫的照片。数据增强 = 把照片旋转 90°、左右翻转、上下翻转……生成 8 张照片。猫还是那只猫，但神经网络看到的是 8 个不同视角。

**代码里的体现**：
```python
# TSProblemDef.py
def augment_xy_data_by_8_fold(problems):
    # 8种变换：原图 + 左右翻 + 上下翻 + 对角线翻 + 旋转90°的各种组合
    dat1 = torch.cat((x, y), dim=2)           # 原版
    dat2 = torch.cat((1 - x, y), dim=2)       # 左右镜像
    dat3 = torch.cat((x, 1 - y), dim=2)      # 上下镜像
    ...
    return torch.cat((dat1, dat2, ..., dat8), dim=0)  # 8倍数据
```

**收益**：
- ✅ 测试时做 augmentation → 同一个题目看 8 个角度，选最好的答案
- ⚠️ 训练时**没做** augmentation → 这是一个可以改进的地方！

---

### 6. POMO（Policy Optimization with Multiple Optima）

**是什么？**
> 一个聪明的训练技巧：对同一个 TSP 实例，同时从**不同的起点**出发走路线，然后互相比较。

**打个比方**：
> 10 个同学同时做同一道 TSP 题，每人从不同的城市开始走。走得最短的路线就是"标准答案"，其他人向这个标准看齐。

**代码里的体现**：
```python
# TSPEnv.py 中 reset()
self.pomo_size = problem_size   # POMO 起点数 = 城市数
# 每个 batch 里有 problem_size 条路线同时走
```

**收益**：
- ✅ 天然有"baseline"（平均分），不需要额外训练一个评价网络
- ✅ 同一个问题得到多种解法，训练更稳定

---

### 7. Encoder（编码器）和 Decoder（解码器）

**是什么？**
> Encoder = 读题（把城市坐标变成向量表示）
> Decoder = 答题（根据已走的城市，预测下一个去哪个）

**打个比方**：
> Encoder = 你读数学题，把文字理解成脑子里的"已知条件"
> Decoder = 你根据已知条件，一步一步写出解题过程

**代码里的体现**：
```python
# TSPModel.py
class TSPModel(nn.Module):
    def __init__(self):
        self.encoder = TSP_Encoder(...)   # 读题
        self.decoder = TSP_Decoder(...)   # 答题

    def pre_forward(self, reset_state):
        self.encoded_nodes = self.encoder(reset_state.problems)  # 编码城市坐标
        self.decoder.set_kv(self.encoded_nodes)                   # 把编码结果给解码器用

    def forward(self, state):
        # Decoder 根据当前位置和已走城市，输出下一步的概率
        probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
        return selected, prob
```

---

### 8. Gap（差距）

**是什么？**
> 你的模型算出来的路线长度，和已知最优路线长度的百分比差距。

**公式**：
```
gap = (你的答案 - 最优答案) / 最优答案 × 100%
```

**代码里的体现**：
```python
# TSPTester_LIB.py
aug_gap = (aug_score - optimal) / optimal * 100
```

**你的 baseline**：
- `avg_aug_gap = 2.33%` → 平均比最优答案多走 2.33%

---

## 🧮 训练流程详细拆解

### Step 1: 出题（TSProblemDef.py）

```python
def get_random_problems(batch_size, problem_size):
    # 生成 batch_size 个随机 TSP 实例
    # 每个实例有 problem_size 个城市，坐标在 [0,1] 范围内
    problems = torch.rand(size=(batch_size, problem_size, 2))
    return problems
```

**结果**：`problems` 的 shape 是 `(64, 100, 2)` → 64 个实例，每个 100 个城市，每个城市 (x,y) 坐标。

---

### Step 2: 环境初始化（TSPEnv.py）

```python
class TSPEnv:
    def __init__(self, problem_size=100, pomo_size=100):
        self.problem_size = 100   # 每个实例有100个城市
        self.pomo_size = 100      # 从100个不同起点同时走

    def reset(self):
        # 清零：还没走任何城市
        self.selected_count = 0
        self.selected_node_list = []   # 已走城市列表
        self.ninf_mask = 全0          # 还没有城市被"禁用"
```

---

### Step 3: 编码城市（TSPModel.py - Encoder）

```python
class TSP_Encoder(nn.Module):
    def forward(self, data):
        # data: (batch, 100, 2) 城市坐标
        embedded_input = self.embedding(data)   # (64, 100, 128) 把坐标变成128维向量
        
        for layer in self.layers:   # 6层 Transformer
            out = layer(out)        # 每层让城市之间的关系更清晰
        
        return out  # (64, 100, 128) 每个城市有了一个"理解后的向量"
```

---

### Step 4: 走路线（TSPModel.py - Decoder + TSPEnv）

```python
# 第1步：从每个城市出发（POMO起点）
selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
# → shape: (64, 100) 64个batch，每个batch有100条路线，分别从不同城市出发

# 第2~100步：根据当前位置预测下一步
while not done:
    # Decoder："我现在在A城市，哪个城市最该去？"
    probs = decoder(encoded_last_node, mask)  # (64, 100, 100) 每个起点对每个城市的倾向
    
    # 选一个城市
    selected = probs.argmax(dim=2)  # 选概率最大的
    
    # 环境更新：标记这个城市已走，不能再去
    env.step(selected)  # 更新 ninf_mask，记录路线
    
    # 走完100个城市就结束
    done = (selected_count == problem_size)
```

---

### Step 5: 计算 Loss（TSPTrainer.py）

```python
# 走完100步后，得到每条路线的总长度（reward = -长度）

# POMO baseline：同一个实例里100条路线的平均分
advantage = reward - reward.mean(dim=1, keepdims=True)
# → 走得比平均分短的 → advantage 为正（好！）
# → 走得比平均分长的 → advantage 为负（差！）

# 把概率取 log，然后乘 advantage
log_prob = prob_list.log().sum(dim=2)   # 每一步选择概率的 log 之和
loss = -advantage * log_prob             # 好路线要增加概率，差路线要减少概率

loss.mean().backward()   # 反向传播：计算怎么调整参数
optimizer.step()         # 实际调整参数
```

**核心思想**：
> 走得好的路线 → 让它的选择概率变大
> 走得差的路线 → 让它的选择概率变小

---

## 📝 带注释的核心代码

### train.py（训练配置文件）

```python
# ========================
# 机器环境配置
# ========================
DEBUG_MODE = False              # 调试模式：True时只跑2轮，方便查bug
USE_CUDA = not DEBUG_MODE       # 是否用 GPU（Mac/无显卡时设为 False）
CUDA_DEVICE_NUM = 0             # 用第几块 GPU

# ========================
# 模型超参数（"大脑"的结构）
# ========================
model_params = {
    'embedding_dim': 128,       # 城市坐标的向量维度。越大"理解力"越强，但计算越慢
    'sqrt_embedding_dim': 128**(1/2),  # 用在注意力计算里的缩放因子
    'encoder_layer_num': 6,     # Encoder 的层数。层数越多，捕捉关系能力越强
    'qkv_dim': 16,              # 注意力机制中每个头的维度
    'head_num': 8,              # 注意力头的数量。8个头 = 同时从8个角度看关系
    'logit_clipping': 10,       # 限制输出概率的极端程度，防止"太自信"
    'ff_hidden_dim': 512,       # Feed Forward 层的隐藏维度
    'eval_type': 'argmax',      # 测试时选概率最大的。训练时自动切换为 softmax 采样
}

# ========================
# 优化器超参数（"教练"的调教方式）
# ========================
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,             # 学习率：每次调整的步长（0.0001）
        'weight_decay': 1e-6     # 权重衰减：防止参数变得太大（正则化）
    },
    'scheduler': {
        'milestones': [3401,],  # 第3401轮时降低学习率
        'gamma': 0.1            # 降到原来的 10%
    }
}

# ========================
# 训练超参数
# ========================
trainer_params = {
    'use_cuda': USE_CUDA,       # 是否用 GPU
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 3800,             # 训练总轮数（从3000续训到3800 = 再训800轮）
    'train_episodes': 100 * 1000,  # 每轮训练多少道题（10万道）
    'train_batch_size': 64,     # 一次做多少道题（64道）
    'logging': {
        'model_save_interval': 100,  # 每100轮存一次模型
        'img_save_interval': 100,    # 每100轮存一次训练曲线图
    },
    'model_load': {
        'enable': True,         # 是否加载旧模型
        'path': './result/saved_tsp100_model2_longTrain',
        'epoch': 3000,          # 加载第3000轮的模型
    }
}

# ========================
# 环境超参数
# ========================
env_params = {
    'problem_size': 100,        # 默认训练100个城市
    'pomo_size': 100,           # POMO起点数 = 城市数
    'mixed_sizes': [100, 150, 200, 250],  # 混合规模训练（新增！）
}
```

---

### TSPModel.py（神经网络核心）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TSPModel(nn.Module):
    """
    TSP 的神经网络模型 = Encoder（读题） + Decoder（答题）
    """
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        
        # Encoder: 把城市坐标变成"理解向量"
        self.encoder = TSP_Encoder(**model_params)
        # Decoder: 根据已走路线预测下一步
        self.decoder = TSP_Decoder(**model_params)
        
        # 保存编码结果，Decoder 每一步都需要用到
        self.encoded_nodes = None

    def pre_forward(self, reset_state):
        """
        编码阶段：一次性把所有城市坐标编码成向量
        这一步在每条路线开始前只做一次
        """
        self.encoded_nodes = self.encoder(reset_state.problems)
        # shape: (batch, 城市数, 128维向量)
        self.decoder.set_kv(self.encoded_nodes)  # 把编码结果存到 Decoder

    def forward(self, state):
        """
        解码阶段：根据当前状态，输出下一步选哪个城市
        
        state: 当前环境状态（在哪、哪些城市已走）
        返回: (selected, prob) - 选的城市索引 + 选它的概率
        """
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.POMO_IDX.size(1)

        if state.current_node is None:
            # ===== 第1步：初始化起点 =====
            # POMO: 从每个城市出发，生成 problem_size 条路线
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            # shape: (batch, pomo) = (64, 100)
            
            prob = torch.ones(size=(batch_size, pomo_size))  # 起点概率都=1
            
            # 把起点城市的编码向量传给 Decoder
            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            self.decoder.set_q1(encoded_first_node)

        else:
            # ===== 第2~N步：根据当前位置预测下一步 =====
            # 获取当前所在城市的编码向量
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            
            # Decoder 输出：对每个未走城市的"想去程度"概率
            probs = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask)
            # shape: (batch, pomo, problem) = (64, 100, 100)

            if self.training or self.model_params['eval_type'] == 'softmax':
                # ===== 训练时：按概率随机采样 =====
                # 好处：探索不同路线，不死脑筋
                while True:
                    # multinomial: 按概率分布随机选1个
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1)
                    selected = selected.squeeze(dim=1).reshape(batch_size, pomo_size)
                    
                    # 拿到选中城市的概率
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected]
                    
                    # 如果概率=0说明有bug，重新采样
                    if (prob != 0).all():
                        break
            else:
                # ===== 测试时：贪心选最大概率 =====
                # 好处：选最稳的路线
                selected = probs.argmax(dim=2)
                prob = None  # 测试时不需要记录概率算 loss

        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    """
    从编码结果中，取出指定城市的向量
    
    encoded_nodes: (batch, 城市数, 128维)
    node_index_to_pick: (batch, pomo) 要取的城市索引
    
    返回: (batch, pomo, 128维) 这些城市的编码向量
    """
    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    # 构造索引，告诉 PyTorch 去哪个位置取
    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    
    # gather: 按索引取值
    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    return picked_nodes


# ========================
# Encoder（编码器）
# ========================
class TSP_Encoder(nn.Module):
    """
    把城市坐标变成高维向量，让模型"理解"城市之间的关系
    
    结构：线性投影 -> 6层 Transformer -> 输出编码向量
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        encoder_layer_num = model_params['encoder_layer_num']

        # 第一层：把2维坐标投影到128维
        self.embedding = nn.Linear(2, embedding_dim)
        
        # 6层 Transformer 编码层
        self.layers = nn.ModuleList([
            EncoderLayer(**model_params) for _ in range(encoder_layer_num)
        ])

    def forward(self, data):
        # data: (batch, 城市数, 2) 原始坐标
        
        embedded_input = self.embedding(data)
        # (batch, 城市数, 128) 每个城市变成了128维向量
        
        out = embedded_input
        for layer in self.layers:
            out = layer(out)  # 每层让城市间关系更清晰
        
        return out


class EncoderLayer(nn.Module):
    """
    单层 Transformer Encoder
    
    核心：Multi-Head Attention（多头注意力）
    - 每个城市"看看"其他城市，决定哪些城市对它重要
    - 8个头 = 同时从8个角度评估关系
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        # 把输入向量变成 Query、Key、Value
        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        # 把多头结果合并回 embedding_dim
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        # Add & Norm + Feed Forward（标准 Transformer 结构）
        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        self.feedForward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        head_num = self.model_params['head_num']

        # === Multi-Head Attention ===
        # Q: "我想知道什么"
        # K: "我能提供什么信息"
        # V: "实际的信息内容"
        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, 8头, 城市数, 16维)

        # 注意力计算：Q 和 K 的相似度决定从 V 取多少信息
        out_concat = multi_head_attention(q, k, v)
        # (batch, 城市数, 8*16=128维)

        multi_head_out = self.multi_head_combine(out_concat)
        # (batch, 城市数, 128)

        # === Add & Norm + Feed Forward ===
        out1 = self.addAndNormalization1(input1, multi_head_out)  # 残差连接
        out2 = self.feedForward(out1)                             # 前馈网络
        out3 = self.addAndNormalization2(out1, out2)              # 再次残差

        return out3


# ========================
# Decoder（解码器）
# ========================
class TSP_Decoder(nn.Module):
    """
    根据当前所在城市，预测下一个去哪个城市
    
    核心思路："我现在的位置 + 出发点的信息" → 看看所有未走城市 → 输出概率
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        head_num = model_params['head_num']
        qkv_dim = model_params['qkv_dim']

        # 两套 Query 投影：一套给起点用，一套给当前位置用
        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        
        # Key 和 Value 来自 Encoder 的输出（所有城市的编码）
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

    def set_kv(self, encoded_nodes):
        """
        预计算 Key 和 Value（来自 Encoder 的城市编码）
        这一步在每条路线前只做一次，之后重复用
        """
        head_num = self.model_params['head_num']
        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        
        # single_head_key 用于最后的概率计算
        self.single_head_key = encoded_nodes.transpose(1, 2)

    def set_q1(self, encoded_q1):
        """
        设置起点城市的 Query（影响整条路线的"风格"）
        """
        head_num = self.model_params['head_num']
        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)

    def forward(self, encoded_last_node, ninf_mask):
        """
        核心推理：当前位置 + 起点信息 → 预测下一步
        
        encoded_last_node: 当前所在城市的编码向量
        ninf_mask: 已经走过的城市标记为 -inf（不能再去）
        
        返回: probs (batch, pomo, 城市数) 每个城市的概率
        """
        head_num = self.model_params['head_num']

        # === Multi-Head Attention ===
        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # 当前位置的 Query

        q = self.q_first + q_last
        # 起点信息 + 当前位置信息 → 综合判断

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # 注意力：看看所有未走城市，输出关注向量

        mh_atten_out = self.multi_head_combine(out_concat)
        # (batch, pomo, 128)

        # === Single-Head Attention: 计算概率 ===
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # 计算和每个城市编码的匹配度
        # (batch, pomo, 128) @ (batch, 128, 城市数) = (batch, pomo, 城市数)

        # 缩放 + 裁剪 → Softmax 输出概率
        score_scaled = score / self.model_params['sqrt_embedding_dim']
        score_clipped = self.model_params['logit_clipping'] * torch.tanh(score_scaled)
        score_masked = score_clipped + ninf_mask  # 已走城市变成 -inf

        probs = F.softmax(score_masked, dim=2)
        return probs


# ========================
# 辅助函数
# ========================
def reshape_by_heads(qkv, head_num):
    """
    把 (batch, n, 8*16)  reshape 成 (batch, 8, n, 16)
    让 PyTorch 能并行处理 8 个注意力头
    """
    batch_s = qkv.size(0)
    n = qkv.size(1)
    
    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    q_transposed = q_reshaped.transpose(1, 2)
    
    return q_transposed  # (batch, head_num, n, key_dim)


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None):
    """
    注意力机制的核心计算
    
    公式: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
    
    通俗理解：
    - Q 问: "我和谁关系好？"
    - K 答: "我的特征是什么"
    - 相似度决定从 V（实际信息）取多少
    """
    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    input_s = k.size(2)

    # Q @ K^T: 计算相似度
    score = torch.matmul(q, k.transpose(2, 3))
    # (batch, head, n, key_dim) @ (batch, head, key_dim, 城市数)
    # = (batch, head, n, 城市数)

    # 除以 sqrt(d) 防止数值爆炸
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    
    # 已走城市变成 -inf（softmax 后概率=0）
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask

    # Softmax 归一化为概率
    weights = nn.Softmax(dim=3)(score_scaled)
    
    # 用概率加权取 Value
    out = torch.matmul(weights, v)
    # (batch, head, n, key_dim)

    # reshape 回标准格式
    out_transposed = out.transpose(1, 2)
    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    
    return out_concat


class Add_And_Normalization_Module(nn.Module):
    """
    Add & Norm: Transformer 的标配
    
    作用：
    1. 残差连接（Add）：新信息 + 原信息 → 防止信息丢失
    2. Layer Norm（Norm）：把向量长度标准化 → 训练更稳定
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        added = input1 + input2           # 残差：保留原始信息
        transposed = added.transpose(1, 2)  # 为了 InstanceNorm 调整维度
        normalized = self.norm(transposed)  # 标准化
        back_trans = normalized.transpose(1, 2)
        return back_trans


class Feed_Forward_Module(nn.Module):
    """
    前馈网络：对注意力输出做进一步处理
    
    结构：线性层(128->512) → ReLU → 线性层(512->128)
    """
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        return self.W2(F.relu(self.W1(input1)))
```

---

### TSPEnv.py（环境模拟）

```python
from dataclasses import dataclass
import torch
from TSProblemDef import get_random_problems, augment_xy_data_by_8_fold

@dataclass
class Reset_State:
    """
    环境重置时的状态：把题目交给模型
    """
    problems: torch.Tensor  # (batch, 城市数, 2) 城市坐标

@dataclass
class Step_State:
    """
    每一步的状态：告诉模型现在在哪、哪些城市已走
    """
    BATCH_IDX: torch.Tensor       # (batch, pomo) batch索引
    POMO_IDX: torch.Tensor        # (batch, pomo) POMO起点索引
    current_node: torch.Tensor = None   # (batch, pomo) 当前所在城市
    ninf_mask: torch.Tensor = None       # (batch, pomo, 城市数) 已走= -inf


class TSPEnv:
    """
    TSP 环境：模拟走路线的过程
    
    核心功能：
    1. load_problems: 加载题目（随机生成或 TSPLIB）
    2. reset: 清零，准备开始
    3. step: 走一步，更新状态
    4. _get_travel_distance: 计算路线总长度
    """
    def __init__(self, **env_params):
        self.env_params = env_params
        self.problem_size = env_params['problem_size']   # 城市数量
        self.pomo_size = env_params['pomo_size']          # POMO 起点数量

    def load_problems(self, batch_size, aug_factor=1):
        """
        生成或加载 batch_size 个 TSP 实例
        
        aug_factor: 数据增强倍数。aug_factor=8 时生成 8 倍数据
        """
        self.batch_size = batch_size
        self.problems = get_random_problems(batch_size, self.problem_size)
        # problems: (batch, 城市数, 2)
        
        if aug_factor > 1:
            if aug_factor == 8:
                # 8倍数据增强：原图 + 各种翻转旋转
                self.batch_size = self.batch_size * 8
                self.problems = augment_xy_data_by_8_fold(self.problems)
        
        # 创建索引张量，方便后续操作
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(
            self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(
            self.batch_size, self.pomo_size)

    def reset(self):
        """
        重置环境：所有状态清零
        
        返回 Reset_State 给 Encoder 编码城市坐标
        """
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.zeros(
            (self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0) 已走城市列表，初始为空
        
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros(
            (self.batch_size, self.pomo_size, self.problem_size))
        # ninf_mask: 0 = 可去, -inf = 已去（softmax后概率=0）

        return Reset_State(self.problems), None, False

    def pre_step(self):
        """
        预步：返回初始状态（还没走任何城市）
        """
        return self.step_state, None, False

    def step(self, selected, lib_mode=False):
        """
        走一步：
        
        selected: (batch, pomo) 选中的城市索引
        lib_mode: 是否用 TSPLIB 的整数距离计算
        
        返回: (new_state, reward, done)
        - new_state: 更新后的状态
        - reward: 走完时返回 -路线长度，没走完返回 None
        - done: 是否走完所有城市
        """
        self.selected_count += 1
        self.current_node = selected
        
        # 把选中的城市加入已走列表
        self.selected_node_list = torch.cat(
            (self.selected_node_list, self.current_node[:, :, None]), dim=2)

        # 更新状态
        self.step_state.current_node = self.current_node
        # 标记这个城市已走：ninf_mask 对应位置设为 -inf
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')

        # 检查是否走完
        done = (self.selected_count == self.problem_size)
        
        if done:
            # 走完了！计算路线长度作为 reward（取负号）
            reward = -self._get_travel_distance(lib_mode=lib_mode)
        else:
            reward = None  # 还没走完，没有 reward

        return self.step_state, reward, done

    def _get_travel_distance(self, lib_mode=False):
        """
        计算每条路线的总旅行距离
        
        原理：
        1. 按已走顺序取出城市坐标
        2. 每个城市到下一个城市的距离求和
        3. 最后还要回到起点
        """
        # gathering_index: 告诉 gather 按什么顺序取坐标
        gathering_index = self.selected_node_list.unsqueeze(3).expand(
            self.batch_size, -1, self.problem_size, 2)
        
        if lib_mode and self.original_node_xy_lib is not None:
            # TSPLIB 模式：用原始坐标 + 整数距离
            base = self.original_node_xy_lib
            if base.dim() == 3 and base.size(0) == 1 and self.batch_size != 1:
                base = base.expand(self.batch_size, -1, -1)
            seq_expanded = base[:, None, :, :].expand(
                self.batch_size, self.pomo_size, self.problem_size, 2)
        else:
            # 训练模式：用随机生成的坐标 + 浮点距离
            seq_expanded = self.problems[:, None, :, :].expand(
                self.batch_size, self.pomo_size, self.problem_size, 2)

        # 按已走顺序取出坐标
        ordered_seq = seq_expanded.gather(dim=2, index=gathering_index)
        
        # roll: 把序列右移，这样 ordered_seq[i] 和 rolled_seq[i] 就是相邻两个城市
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        
        # 欧氏距离
        segment_lengths_raw = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        
        if lib_mode:
            # TSPLIB 整数距离规则
            ewt = self.edge_weight_type or 'EUC_2D'
            if ewt == 'CEIL_2D':
                segment_lengths = torch.ceil(segment_lengths_raw)
            elif ewt == 'EUC_2D':
                segment_lengths = torch.floor(segment_lengths_raw + 0.5)
            else:
                segment_lengths = segment_lengths_raw
        else:
            segment_lengths = segment_lengths_raw  # 训练用浮点距离
        
        # 所有段长度求和 = 总路线长度
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        
        return travel_distances
```

---

### TSPTrainer.py（训练控制器）

```python
import torch
from TSPEnv import TSPEnv as Env
from TSPModel import TSPModel as Model
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils.utils import *

class TSPTrainer:
    """
    训练器：控制整个训练流程
    
    核心循环（每轮 epoch）：
    1. 生成 batch 个随机 TSP 实例
    2. 让模型走路线
    3. 计算 Loss（好路线增加概率，差路线减少）
    4. 反向传播更新模型参数
    5. 重复直到 episode 数达标
    """
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        # 保存配置
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params
        
        # 设置日志
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        
        # 设置设备（CPU/GPU）
        USE_CUDA = trainer_params['use_cuda']
        if USE_CUDA:
            device = torch.device('cuda', trainer_params['cuda_device_num'])
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        
        # 初始化三大件
        self.model = Model(**model_params)          # 神经网络
        self.env = Env(**env_params)                # 环境
        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])
        
        # === 加载已有模型（续训）===
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
        
        # === 混合规模训练 ===
        self.mixed_sizes = self.env_params.get(
            'mixed_sizes', [self.env_params['problem_size']])
        
        self.time_estimator = TimeEstimator()

    def run(self):
        """
        主训练循环：跑 epochs 轮
        """
        for epoch in range(self.start_epoch, self.trainer_params['epochs'] + 1):
            # 学习率衰减
            self.scheduler.step()
            
            # 训练一轮
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            
            # 定期保存模型和训练曲线
            if epoch % model_save_interval == 0:
                torch.save(checkpoint_dict, f'{result_folder}/checkpoint-{epoch}.pt')
    
    def _train_one_epoch(self, epoch):
        """
        训练一轮：直到 episode 数达标（默认 10万道题）
        """
        score_AM = AverageMeter()  # 记录平均得分（路线长度）
        loss_AM = AverageMeter()   # 记录平均 Loss
        
        train_num_episode = self.trainer_params['train_episodes']  # 10万
        episode = 0
        
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            
            # 训练一个 batch
            avg_score, avg_loss, actual_bs = self._train_one_batch(batch_size)
            score_AM.update(avg_score, actual_bs)
            loss_AM.update(avg_loss, actual_bs)
            
            episode += actual_bs
        
        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        """
        训练一个 batch 的核心逻辑
        """
        import random
        
        # === 混合规模：随机选城市规模 ===
        problem_size = random.choices(self.mixed_sizes, weights=[3, 2, 1, 1])[0]
        # weights: 100城市训练更频繁(3)，250城市少一些(1)
        
        if problem_size != self.env.problem_size:
            self.env.problem_size = problem_size
            self.env.pomo_size = problem_size
        
        # === 动态 batch_size：大城市用小 batch 省显存 ===
        base_batch = self.trainer_params['train_batch_size']
        scale = (100 / problem_size) ** 2
        adjusted_batch = max(16, int(base_batch * scale))
        batch_size = min(adjusted_batch, batch_size)
        
        # === 准备阶段 ===
        self.model.train()                    # 切换到训练模式（启用 dropout 等）
        self.env.load_problems(batch_size)    # 生成随机题目
        reset_state, _, _ = self.env.reset()  # 环境清零
        self.model.pre_forward(reset_state)   # Encoder 编码城市
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # 记录每一步选择的概率（后续算 Loss 用）
        
        # === POMO Rollout：让模型走完整条路线 ===
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)   # 模型预测下一步
            state, reward, done = self.env.step(selected)  # 环境更新
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            # 把这一步的概率存起来
        
        # === 计算 Loss ===
        # reward: (batch, pomo) 每条路线的负长度
        
        # POMO baseline：同一个实例里 pomo 条路线的平均分
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # 比平均分好的 → 正数；差的 → 负数
        
        # 计算每条路线的总概率（log）
        log_prob = prob_list.log().sum(dim=2)
        # size: (batch, pomo)
        
        # Loss：好路线要增加概率，差路线要减少
        # -advantage * log_prob：
        #   advantage > 0（好路线） → 想让 loss 小 → 需要 log_prob 大（概率大）
        #   advantage < 0（差路线） → 想让 loss 小 → 需要 log_prob 小（概率小）
        loss = -advantage * log_prob
        loss_mean = loss.mean()
        
        # 得分：取每条路线最好的结果
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()  # 负号变正数（长度）
        
        # === 反向传播更新参数 ===
        self.model.zero_grad()    # 清空旧梯度
        loss_mean.backward()      # 计算新梯度（告诉每个参数该怎么调）
        self.optimizer.step()     # 实际调整参数
        
        return score_mean.item(), loss_mean.item(), batch_size
```

---

### TSPTester_LIB.py（测试器，带多采样 + 2-opt）

```python
class TSPTester_LIB:
    """
    测试器：用 TSPLIB 真实题目考模型
    
    核心流程：
    1. 读取 .tsp 文件
    2. 归一化坐标到 [0,1]
    3. 数据增强 8 倍
    4. 多采样推理（num_samples 次）
    5. 可选 2-opt 后处理
    6. 和最优解比较算 gap
    """
    
    def _test_one_instance(self, nodes_xy_normalized, coords_orig, ew_type):
        """
        测试一个 TSP 实例
        
        参数：
        - nodes_xy_normalized: 归一化后的坐标 (1, N, 2)
        - coords_orig: 原始坐标 (N, 2)
        - ew_type: 距离类型（EUC_2D 或 CEIL_2D）
        
        返回：(no_aug_score, aug_score)
        """
        # 1. 读取参数
        aug_factor = 8 if augmentation_enable else 1
        num_samples = self.tester_params.get('num_samples', 1)
        enable_2opt = self.tester_params.get('enable_2opt', False)
        
        # 2. 数据增强 8 倍
        problems = augment_xy_data_by_8_fold(nodes_xy_normalized)
        # shape: (8, N, 2)
        
        # 3. 初始化环境
        env = Env(problem_size=problem_size, pomo_size=problem_size)
        env.batch_size = 8  # 8个增强版本
        env.problems = problems.to(device)
        # ... 设置其他环境参数 ...
        
        # 4. 多采样推理
        all_sample_best = []
        
        # Encoder 只算一次！（所有采样共享）
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        
        for sample_id in range(num_samples):
            # 前 N-1 次用随机采样，最后1次用贪心
            eval_type = 'softmax' if sample_id < num_samples - 1 else 'argmax'
            self.model.model_params['eval_type'] = eval_type
            
            # 重置环境动态状态（不清空 Encoder 结果）
            env.selected_count = 0
            env.current_node = None
            env.selected_node_list = ...
            
            # 走完整条路线
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = env.step(selected, lib_mode=True)
            
            tour_lengths = -reward  # 取反得到实际长度
            
            # 5. 可选 2-opt
            if enable_2opt:
                # 对每条路线做 2-opt 局部优化
                for b in range(8):
                    tour = env.selected_node_list[b, best_idx].cpu().tolist()
                    optimized = _two_opt(tour, dist_matrix)
                    opt_len = _tour_length(optimized, dist_matrix)
                    all_sample_best.append(opt_len)
            else:
                all_sample_best.append(tour_lengths.min(dim=1).values.cpu())
        
        # 6. 汇总结果
        all_sample_best = torch.stack(all_sample_best, dim=0)
        no_aug_score = all_sample_best[:, 0].min().item()  # 无增强的最优
        aug_score = all_sample_best.min().item()            # 全局最优
        
        return no_aug_score, aug_score
```

---

## 📊 你的实验数据解读

| 实验 | avg_aug_gap | 说明 | 解读 |
|------|------------|------|------|
| Baseline | **2.33%** | 原模型 + aug8 | 100城市训练，大实例（pr299）表现差 |
| 多采样×8 | **2.27%** | +num_samples=8 | 多采样几乎没效果 |
| 多采样+2opt | **0.67%** | +2-opt | 2-opt 是决定性因素 |

**关键发现**：
- `pr299`（299城市）gap = 13.98% → 模型**完全没见过**这么大的问题
- `eil101`（101城市）gap = 0.00% → 小实例已经**完美**
- 所以混合规模训练**方向是对的**，但还不够

---

## 🎯 下一步改进建议（不用 2-opt）

### 立即能做的（1天）

1. **训练时加数据增强**
   - 改 `load_problems` 让训练时也做 augmentation
   - 预期：让小实例更稳

2. **增大 num_samples**
   - 从 8 改到 32 或 64
   - 预期：多采样可能有边际收益

3. **Temperature 调参**
   - 测试时加温度参数控制采样的"随机程度"
   - 预期：找到比 argmax 更好的平衡点

### 中期（等 GPU 跑）

4. **续训更多 epoch**（5000+）
5. **更大 batch / 梯度累积**
6. **Cosine Annealing 学习率**

### 长期（如果时间够）

7. **增大模型**（256维 + 8层）→ 需从头训练

---

## ❓ FAQ

**Q: 为什么不用 2-opt 差距这么大？**
> A: 神经网络生成的是"大概不错的路线"，但可能有交叉。2-opt 专门消除交叉，一消就是百分之几的提升。没它就得靠模型本身足够强。

**Q: 混合规模训练是什么意思？**
> A: 以前只做100城市的题，现在100/150/200/250都练。就像只练小学题去考大学——肯定不行。

**Q: 为什么 POMO baseline 比 Critic 好？**
> A: POMO 的 baseline 是"同一道题100个同学一起做取平均"，方差天然小。学出来的 Critic 可能反而看错。

**Q: 我的 Mac 能跑训练吗？**
> A: 可以用 CPU 跑（把 USE_CUDA 改成 False），但会很慢。建议用学校的 GPU 服务器。

---

## 📝 代码修改记录

| 改动 | 文件 | 说明 |
|------|------|------|
| 混合规模 | train.py | `mixed_sizes: [100,150,200,250]` |
| 权重采样 | TSPTrainer.py | `random.choices(weights=[3,2,1,1])` |
| 动态 batch | TSPTrainer.py | 250城市缩到16 |
| 多采样 | test.py | `--num_samples 8` |
| 2-opt | TSPTester_LIB.py | `--enable_2opt true` |
| 续训 | train.py | `epochs: 3800, model_load.epoch: 3000` |

---

_文档生成时间：2026-04-22 by 塔塔air 🌬️_
_有任何问题随时问，不用怕"问得太基础"——从零开始完全没问题。_
