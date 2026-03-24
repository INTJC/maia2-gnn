# Maia2-GNN

这是一个基于 `maia2-main` 重构的独立新项目，将原先的 CNN 主干替换为 GNN，并采用图结构表示棋盘状态。

## 设计目标

- 棋盘表示：每个格点一个节点（64 节点），走法作为边；
- 边特征：参考 `AlphaGateau-DoubleGNN` 思路，编码合法性、位移、升变类型、走子类型；
- 输出任务：保留原 Maia2 的三路输出（走子策略、side-info、多任务 value）。

## 安装

```bash
pip install -e .
```

## 快速示例

```python
import chess
import torch

from maia2_gnn.utils import get_all_possible_moves, create_elo_dict
from maia2_gnn.graph_utils import board_to_graph
from maia2_gnn.model import Maia2GNNModel, GNNConfig

all_moves = get_all_possible_moves()
all_moves_dict = {m: i for i, m in enumerate(all_moves)}
elo_dict = create_elo_dict()

cfg = GNNConfig()
model = Maia2GNNModel(num_moves=len(all_moves), elo_vocab_size=len(elo_dict), cfg=cfg)

board = chess.Board()
graph = board_to_graph(board, all_moves, all_moves_dict)

batch = graph.to("cpu")
with torch.no_grad():
    policy_logits, side_logits, value = model(batch)
```

## 训练 / 验证 / 测试（完整流程）

已提供完整 pipeline：

- 从 PGN 解析样本（按 ply 范围过滤）；
- 自动划分 train/val/test（默认 8:1:1）；
- 每个 epoch 执行训练 + 验证；
- 保存 `best.pt`（按验证集 move_acc）和 `last.pt`；
- 支持 early stopping（默认 patience=5）；
- 最后用最佳模型在测试集评估并输出 `train_report.json`。

运行示例：

```bash
python -m maia2_gnn.train --pgn_path "maia2_gnn/example_training_data.pgn" --save_dir "./saves"

# 用 PGN 做训练/验证，用 CSV 做最终测试
python -m maia2_gnn.train --pgn_path "maia2_gnn/example_training_data.pgn" --test_csv_path "maia2_gnn/rapid_sample_data.csv" --save_dir "./saves"
# 如需快速调试，可限制测试样本数量
python -m maia2_gnn.train --pgn_path "maia2_gnn/example_training_data.pgn" --test_csv_path "maia2_gnn/rapid_sample_data.csv" --test_max_rows 5000 --save_dir "./saves"
```

推荐默认参数（已经内置）：

- `epochs=20`
- `batch_size=128`
- `lr=3e-4`
- `patience=5`
- `val_ratio=0.1`
- `test_ratio=0.1`

## 一键运行脚本

如果你希望直接运行单个脚本（并可修改各种路径），使用：

```bash
python run_full_pipeline.py
```

可覆盖任意路径和参数，例如：

```bash
python run_full_pipeline.py --pgn_path "maia2_gnn/example_training_data.pgn" --test_csv_path "maia2_gnn/rapid_sample_data.csv" --save_dir "./saves/my_run" --epochs 30 --device cpu
```
