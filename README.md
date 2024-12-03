# CSCE-642: Adapting Insights from DreamerV3 and REDQ into MuZero
## Setup
This project is based on the framework [Stoix](https://github.com/EdanToledo/Stoix/tree/main): Distributed Single-Agent Reinforcement Learning End-to-End in JAX, developed by Edan Toledo. 

```bash
git clone git@github.com:Your-Cheese/Stoix.git
cd Stoix
pip install -e .
```

## Running

### Ensemble of learners
To execute Cartpole, please type command below. Cartpole is the default environment, so there is no need to specify it.
```bash
python stoix/systems/search/ff_mz_ens.py system=search/ff_mz_ens_cp arch=anakin_ens_cp
```

To execute Breakout, please type command below.
```bash
python stoix/systems/search/ff_mz_ens.py system=search/ff_mz_ens_bo arch=anakin_ens_bo env=gymnax/breakout
```
To adjust number of learners, please modify `num_learners` in respective configuration file in `configs/system/search` that starts with `ff_mz_ens_`.

## Acknowledgments

```bibtex
@misc{toledo2024stoix,
    title={Stoix: Distributed Single-Agent Reinforcement Learning End-to-End in JAX},
    doi = {10.5281/zenodo.10916257},
    author={Edan Toledo},
    month = apr,
    year = {2024},
    url = {https://github.com/EdanToledo/Stoix},
}
```