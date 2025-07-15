# Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank and Feature Selection

[![Source Code License](https://img.shields.io/badge/license-GPL-blueviolet)](https://github.com/afrakilic/BTN-Kernel-Machines/blob/main/LICENSE)


This repository contains the source code used to produce the results obtained in [Interpretable Bayesian Tensor Network Kernel Machines with Automatic Rank and Feature Selection](https://arxiv.org/abs/2409.12789) submitted to [Journal of Machine Learning Research](https://www.jmlr.org/).

In this work we propose a fully probabilistic framework that uses sparsity-inducing hierarchical priors on Tensor Network factors to automatically infer tensor rank and feature dimensions, while also identifying the most relevant features for prediction, thereby enhancing model interpretability.

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{
}
```

---

## Installation



Make sure you have Python **3.10.16** installed.

Install dependencies:

```bash
git clone https://github.com/afrakilic/BTN-Kernel-Machines
cd BTN-Kernel-Machines
```

and then install the required packages by, e.g., running

```bash
pip install -r requirements.txt
```

### Structure

The repository code is structured in the following way

- **`agents`** contains the classes defined for RL agents.
- **`data`** contains weather disturbance data.
- **`greenhouse`** contains the model and environments classes for the greenhouse system.
- **`mpcs`** contains the classes for all mpc controllers.
- **`sims/configs`** contains configuration files for simulations.
- **`utils`** contains plotting and evalation scripts used to generate images and data used in Reinforcement Learning-Based Model Predicitive Control for Greenhouse Climate Control
- **`nominal_greenhouse.py`** simulates the nominal mpc controller.
- **`sample_greenhouse.py`** simulates the sample based mpc controller.
- **`q_learning_greenhouse.py`** trains the RL-based mpc controller.
- **`train_ddpg.py`** trains the DDPG-based RL controller.
- **`visualization.py`** vizualizes data saved from simulations.
## License

The repository is provided under the GNU General Public License. See the [LICENSE](https://github.com/afrakilic/BTN-Kernel-Machines/blob/main/LICENSE) file included with this repository.

---

## Author

[Afra KILIC](https://www.tudelft.nl/staff/h.a.kilic/), PhD Candidate [H.A.Kilic@tudelft.nl | hafra.kilic@gmail.com]

> [Delft Center for Systems and Control](https://www.tudelft.nl/en/3me/about/departments/delft-center-for-systems-and-control/) in [Delft University of Technology](https://www.tudelft.nl/en/)

> This research is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (with project number VI.Vidi.213.017) which is financed by the Dutch Research Council (NWO).

Copyright (c) 2025 Afra Kilic.

Copyright notice: Technische Universiteit Delft hereby disclaims all copyright interest. 