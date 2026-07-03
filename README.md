# D2DS

[中文说明](#中文说明) | [English README](#english-readme)

## 中文说明

D2DS 是一个面向二维材料缺陷体系的图神经网络研究代码库。仓库包含缺陷结构数据、训练与评估脚本、已保存的 checkpoint/result，以及用于缺陷形成能等能量性质预测的 DimeNet++ 风格模型实现。项目主要围绕二维材料中的空位、替位和多缺陷体系，比较从头训练、预训练/微调以及 MACE、DimeNet++/DPP 等模型设置。

### 项目结构

```text
.
├── ID2S/          # D2DS 主数据集实验，包含 D2DS.csv、预训练、主训练和消融脚本
├── 2DMD/          # 多个二维材料缺陷数据集的训练与五折实验脚本
├── ABZ/           # AB2Z4 缺陷体系实验，包含不同训练比例、all/compound 切分和微调设置
├── Imp2D/         # Imp2D 数据库实验，基于 database.json 训练缺陷预测模型
├── Dual-defect/   # 双缺陷标签数据
├── common/        # 训练、验证、checkpoint、loss、图构造等通用工具
├── model/         # defect_dpp 主模型及 MACE、DimeNet++ 等对照模型实现
├── LICENSE
└── README.md
```

### 主要功能

- 使用 PyTorch 和 PyTorch Geometric 构建周期性二维材料缺陷图。
- 支持缺陷形成能/相关能量指标的监督学习训练与测试。
- `defect_dpp` 模型将缺陷位点贡献与邻域结构贡献结合，用于描述空位和替位缺陷。
- 多个实验目录提供 `all` 随机切分、`compound` 主体材料切分、预训练、微调、消融和基线模型对比。
- 仓库中保留了部分训练结果和 checkpoint，便于复现实验或继续分析。

### 环境依赖

本项目是研究脚本式代码库，暂未提供统一的 `requirements.txt`。运行前请根据你的 CUDA/CPU 环境安装 PyTorch 与 PyTorch Geometric 生态依赖。代码中常用依赖包括：

- Python 3.x
- PyTorch
- PyTorch Geometric
- torch-scatter
- pandas
- numpy
- scikit-learn
- tqdm

示例环境创建流程：

```bash
conda create -n d2ds python=3.10
conda activate d2ds
# Install PyTorch and PyTorch Geometric according to your CUDA/CPU setup first.
pip install pandas numpy scikit-learn tqdm
```

### 快速开始

```bash
git clone https://github.com/GoodZhenLi/D2DS.git
cd D2DS
```

不同实验脚本使用相对路径读取数据，建议先进入对应目录再运行：

```bash
cd ID2S
python main.py
```

其他常用入口：

```bash
# ID2S 预训练
cd ID2S
python pretrained.py

# 2DMD 多数据集缺陷实验
cd 2DMD
python main.py

# AB2Z4 缺陷体系实验
cd ABZ
python fine_tune_from-scratch.py

# Imp2D 数据库实验
cd Imp2D
python main.py
```

运行前建议检查各脚本底部的超参数、随机种子、训练轮数、模型名称、数据切分方式和输出目录。训练结果通常保存为 `result*/` 下的 `.pkl` 文件，模型权重保存到 `checkpoint*/` 目录。

### 数据说明

仓库中的主要数据文件包括：

- `ID2S/D2DS.csv`
- `2DMD/dataset/*.csv`
- `ABZ/AB2Z4_defect.csv`
- `ABZ/chemical_potential.csv`
- `Imp2D/database.json`
- `Dual-defect/dual_defect_label1.csv`

数据字段会在各目录的 `dataset.py` 中被解析为 PyTorch Geometric `Data` 对象，包括原子位置、原子序数/电荷、晶胞、缺陷位点、边信息以及预测目标。

### 备注

- 这些脚本更接近论文实验代码，而不是已封装的软件包；部分路径、模型选择和训练配置是硬编码的。
- 如果只想复现实验，请优先保持目录结构不变，并在对应实验目录中运行脚本。
- 如果要迁移到自己的数据，请参考各目录的 `dataset.py` 字段格式和 `common/graph_construction.py` 的图构造逻辑。

### 许可证

本项目基于 MIT License 发布，详见 [LICENSE](LICENSE)。

---

## English README

D2DS is a graph-neural-network research codebase for defect systems in two-dimensional materials. The repository contains defect-structure datasets, training/evaluation scripts, saved checkpoints/results, and a DimeNet++-style model implementation for predicting defect formation energies and related energy properties. The experiments focus on vacancies, substitutions, and multi-defect systems in 2D materials, with comparisons among training from scratch, pretraining/fine-tuning, MACE, DimeNet++/DPP, and the project-specific `defect_dpp` model.

### Repository Layout

```text
.
├── ID2S/          # Main D2DS dataset experiments, including D2DS.csv, pretraining, training, and ablation scripts
├── 2DMD/          # Multi-dataset 2D-material defect experiments with fold-based evaluation scripts
├── ABZ/           # AB2Z4 defect experiments with train-ratio sweeps, all/compound splits, and fine-tuning settings
├── Imp2D/         # Imp2D database experiments based on database.json
├── Dual-defect/   # Dual-defect label data
├── common/        # Shared training, validation, checkpointing, loss, and graph-construction utilities
├── model/         # Main defect_dpp model and baseline model implementations such as MACE and DimeNet++
├── LICENSE
└── README.md
```

### Features

- Builds periodic 2D-material defect graphs with PyTorch and PyTorch Geometric.
- Trains supervised models for defect formation energy and related energy-property prediction.
- Combines defect-site and neighboring-structure contributions in the `defect_dpp` model to describe vacancy and substitution defects.
- Provides experiment scripts for random `all` splits, host-level `compound` splits, pretraining, fine-tuning, ablations, and baseline comparisons.
- Keeps selected checkpoints and result files in the repository for reproduction and downstream analysis.

### Environment

This repository is organized as research scripts and does not currently provide a single `requirements.txt`. Install PyTorch and the PyTorch Geometric stack according to your CUDA/CPU environment before running the experiments. Common dependencies include:

- Python 3.x
- PyTorch
- PyTorch Geometric
- torch-scatter
- pandas
- numpy
- scikit-learn
- tqdm

Example setup:

```bash
conda create -n d2ds python=3.10
conda activate d2ds
# Install PyTorch and PyTorch Geometric according to your CUDA/CPU setup first.
pip install pandas numpy scikit-learn tqdm
```

### Quick Start

```bash
git clone https://github.com/GoodZhenLi/D2DS.git
cd D2DS
```

Most experiment scripts read data through relative paths, so run each entry point from its own directory:

```bash
cd ID2S
python main.py
```

Other useful entry points:

```bash
# ID2S pretraining
cd ID2S
python pretrained.py

# 2DMD multi-dataset defect experiments
cd 2DMD
python main.py

# AB2Z4 defect-system experiments
cd ABZ
python fine_tune_from-scratch.py

# Imp2D database experiments
cd Imp2D
python main.py
```

Before running, check the hyperparameters, random seeds, epoch counts, model names, data-splitting settings, and output directories near the bottom of each script. Results are usually saved as `.pkl` files under `result*/`, while model weights are saved under `checkpoint*/`.

### Data

Main data files include:

- `ID2S/D2DS.csv`
- `2DMD/dataset/*.csv`
- `ABZ/AB2Z4_defect.csv`
- `ABZ/chemical_potential.csv`
- `Imp2D/database.json`
- `Dual-defect/dual_defect_label1.csv`

Each directory's `dataset.py` parses these files into PyTorch Geometric `Data` objects, including atomic positions, atomic numbers/charges, cells, defect sites, graph edges, and prediction targets.

### Notes

- The code is closer to paper/research experiment scripts than to a packaged library; some paths, model choices, and training settings are hard-coded.
- To reproduce the existing experiments, keep the repository layout unchanged and run scripts from the corresponding experiment directory.
- To adapt the code to your own data, start from the field formats in each `dataset.py` and the graph-building logic in `common/graph_construction.py`.

### License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
