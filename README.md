# CSCE-642: Adapting Insights from DreamerV3 and REDQ into MuZero
## Setup
This project is based on the framework [Stoix](https://github.com/EdanToledo/Stoix/tree/main): Distributed Single-Agent Reinforcement Learning End-to-End in JAX, developed by Edan Toledo.

```bash
git clone git@github.com:Your-Cheese/Stoix.git
cd Stoix
pip install -e .
```

## Running

### Transformation Functions
ff_mz.py has been set to execute the default config located in [experiments](stoix/configs/experiments).
Options for selecting different transformation functions were implemented into the config for ff_mz. To use them, simply modify the transform_method and transform_function fields to one of the options listed in the comment in [ff_mz.yaml](stoix/configs/system/search/ff_mz.yaml#L29).
For example, to run MuZero with the unbiased transformation method and the symlog function,

```bash
python stoix/systems/search/ff_mz.py system.transform_method=unbiased system.transform_function=symlog
```

In addition, default hyperparameters will be automatically set based on the environment for those with configs in the experiments directory.

```bash
python stoix/systems/search/ff_mz.py env=gymnax/breakout
```

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
We cannot thank the author of the Stoix enough for providing a solid framework for Reinforcement Learning research.
If you use Stoix in your project, please cite:
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
