# Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank Selection
 
[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTN-Kernel-Machines/blob/main/LICENSE)
 
Code for the paper [Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank Selection](https://arxiv.org/abs/2507.11136) (submitted to the SIAM/ASA Journal on Uncertainty Quantification).
 
BTN-Kernel machines place sparsity-inducing hierarchical priors on the factors of a CP-decomposed
weight tensor. Variational inference then infers the tensor rank and the feature dimensions
automatically and gives a closed-form predictive distribution.
 
## Installation
 
Python 3.10 was used.
 
```bash
git clone https://github.com/afrakilic/BTN-Kernel-Machines
cd BTN-Kernel-Machines
pip install -r requirements.txt
```
 
## Repository structure
 
```
btnkm/              the model
  model.py            BTNKM class: train, predict, mc_predict
  utils.py            feature map and linear-algebra helpers
  data.py             data loading, train/test split, NLL and coverage metrics
experiments/        one script per table / figure of the paper
data/               UCI data sets
results/            saved results and figures
T-KRR-SIAM/         T-KRR and GP baselines (MATLAB)
```
 
## Usage
 
```python
from btnkm import BTNKM
 
model = BTNKM().train(X_train, y_train, M=20, R=25,
                      a0=1e-3, b0=1e-3, c0=1e-5, d0=1e-6, g0=1e-6, h0=1e-6, max_iter=50)
mean, std = model.predict(X_test)        # Student's-t predictive, nu = 2 * model.a
print(model.rank)                        # inferred rank
```
 
## Reproducing the paper
 
Run the scripts from any folder; each one prints its table and/or saves its figure to `results/`.
 
| Paper                 | Script                                                          |
|-----------------------|-----------------------------------------------------------------|
| Figure 2              | `python experiments/figure2_synthetic.py`                       |
| Tables 1 and 2        | `python experiments/table1_table2_max_rank.py`                  |
| Figure 3              | `python experiments/figure3_convergence.py`                     |
| Figure 4 and Table 3  | `python experiments/figure4_table3_feature_precision.py`        |
| Tables 4 and 5 (regression) | `python experiments/table4_table5_regression.py`, then `python experiments/table4_table5_summary.py` |
| Table 4 (classification)    | `python experiments/table4_classification.py spambase` (or `adult`) |
 
`results/regression_runs.csv` and `results/regression_runtimes.csv` contain the results
used in the paper, so `table4_table5_summary.py` can be run directly. The GP baseline
predictions are read from `T-KRR-SIAM/<dataset>_all_runs_gp.csv`.
 
All experiments use fixed random seeds. They were run on a MacBook Pro (Apple M2 Pro, 16 GB,
macOS 15.5); other operating systems, hardware or library versions may give slightly
different numbers. The Adult experiment takes about 5.5 hours.
 
## Results
 
UCI benchmarks, 10 random 90/10 splits, mean ± standard deviation over the splits. Times are per
split and cover training and prediction; the Monte Carlo check is not included.
 
| Data set | RMSE | Analytic NLL | MC NLL | Analytic cov. (%) | MC cov. (%) | Effective rank | Time per split (s) |
|----------|------|--------------|--------|-------------------|-------------|----------------|--------------------|
| Airfoil  | 1.721 ± 0.177 | 1.939 ± 0.100 | 1.940 ± 0.100 | 94.8 ± 1.2 | 94.9 ± 1.1 | 9.4 ± 0.8 | 3.34 ± 0.66 |
| Concrete | 4.977 ± 0.475 | 3.032 ± 0.139 | 3.024 ± 0.129 | 91.4 ± 3.0 | 91.3 ± 2.9 | 8.2 ± 0.6 | 6.20 ± 0.72 |
| Energy   | 0.413 ± 0.059 | 0.539 ± 0.135 | 0.543 ± 0.134 | 96.1 ± 2.0 | 96.4 ± 2.2 | 6.5 ± 0.7 | 4.23 ± 1.63 |
| Yacht    | 0.400 ± 0.175 | 0.339 ± 0.390 | 0.408 ± 0.499 | 95.5 ± 3.6 | 95.2 ± 4.4 | 4.6 ± 0.7 | 3.32 ± 2.83 |
 
| Data set | Misclassification | NLL | Effective rank | Time per split (s) |
|----------|-------------------|-----|----------------|--------------------|
| Spambase | 0.094 ± 0.083 | 0.213 ± 0.110 | 2.3 ± 0.8 | 49.54 ± 14.16 |
| Adult    | 0.145 ± 0.005 | 0.332 ± 0.007 | 4.1 ± 0.8 | 1954.99 ± 386.01 |
 
On Spambase, two of the ten splits converge to a rank-1 solution, which raises the mean and
standard deviation of the error; on the remaining eight splits, the average misclassification
error is 0.053.
 
The GP baseline (RMSE, NLL and coverage) is printed by `table4_table5_summary.py` next to these
results.
 
## Citation
 
```bibtex
@misc{kilic2025interpretablebayesiantensornetwork,
      title={Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank Selection},
      author={Afra Kilic and Kim Batselier},
      year={2025},
      eprint={2507.11136},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2507.11136},
}
```
 
## License and author
 
GNU General Public License, see [LICENSE](LICENSE).
 
[Afra Kilic](https://www.tudelft.nl/staff/h.a.kilic/), PhD candidate, [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/), Delft University of Technology (H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com).
This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (project number VI.Vidi.213.017), financed by the Dutch Research Council (NWO).
 
Copyright (c) 2025 Afra Kilic. Technische Universiteit Delft hereby disclaims all copyright interest.