# D2DS

D2DS is a graph-neural-network research codebase for defect systems in two-dimensional materials. The repository contains defect-structure datasets, training and evaluation scripts, saved checkpoints/results, and a DimeNet++-style model implementation for predicting defect formation energies and related energy properties.

The experiments focus on vacancies, substitutions, and multi-defect systems in 2D materials, with comparisons among training from scratch, pretraining/fine-tuning, MACE, DimeNet++/DPP, and the project-specific `defect_dpp` model.

## Repository Layout

```text
.
|-- ID2S/          # Main D2DS dataset experiments, including D2DS.csv, pretraining, training, and ablation scripts
|-- 2DMD/          # Multi-dataset 2D-material defect experiments with fold-based evaluation scripts
|-- ABZ/           # AB2Z4 defect experiments with train-ratio sweeps, all/compound splits, and fine-tuning settings
|-- Imp2D/         # Imp2D database experiments based on database.json
|-- Dual-defect/   # Dual-defect label data
|-- common/        # Shared training, validation, checkpointing, loss, and graph-construction utilities
|-- model/         # Main defect_dpp model and baseline model implementations such as MACE and DimeNet++
|-- LICENSE
`-- README.md
```

## Features

- Builds periodic 2D-material defect graphs with PyTorch and PyTorch Geometric.
- Trains supervised models for defect formation energy and related energy-property prediction.
- Combines defect-site and neighboring-structure contributions in the `defect_dpp` model to describe vacancy and substitution defects.
- Provides experiment scripts for random `all` splits, host-level `compound` splits, pretraining, fine-tuning, ablations, and baseline comparisons.
- Keeps selected checkpoints and result files in the repository for reproduction and downstream analysis.

## Environment

This repository is organized as research scripts and does not currently provide a single `requirements.txt`. Install PyTorch and the PyTorch Geometric stack according to your CUDA/CPU environment before running the experiments.

Common dependencies include:

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

## Quick Start

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

## Data

Main data files include:

- `ID2S/D2DS.csv`
- `2DMD/dataset/*.csv`
- `ABZ/AB2Z4_defect.csv`
- `ABZ/chemical_potential.csv`
- `Imp2D/database.json`
- `Dual-defect/dual_defect_label1.csv`

Each directory's `dataset.py` parses these files into PyTorch Geometric `Data` objects, including atomic positions, atomic numbers/charges, cells, defect sites, graph edges, and prediction targets.

## Notes

- The code is closer to paper/research experiment scripts than to a packaged library; some paths, model choices, and training settings are hard-coded.
- To reproduce the existing experiments, keep the repository layout unchanged and run scripts from the corresponding experiment directory.
- To adapt the code to your own data, start from the field formats in each `dataset.py` and the graph-building logic in `common/graph_construction.py`.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
