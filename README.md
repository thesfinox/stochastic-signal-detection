# Stochastic Signal Detection (SSD)

[Harold Erbin](mailto:harold.erbin@cea.fr), [Riccardo Finotello](mailto:riccardo.finotello@cea.fr), [Bio Wahabou Kpera](mailto:wahaboukpera@gmail.com), [Vincent Lahoche](mailto:vincent.lahoche@cea.fr), [Dine Ousmane Samary](mailto:dine.ousmanesamary@cipma.uac.bj)

[![arXiv](https://img.shields.io/badge/arxiv-2023.07499-red)](https://arxiv.org/abs/2310.07499)
[![github](https://img.shields.io/badge/github-stochastic--signal--detection-blue?logo=github)](https://github.com/thesfinox/stochastic-signal-detection)

Signal detection is one of the main challenges of data science.
As it often happens in data analysis, the signal in the data may be corrupted by noise.
There is a wide range of techniques aimed at extracting the relevant degrees of freedom from data.
However, some problems remain difficult.
It is notably the case of signal detection in almost continuous spectra when the signal-to-noise ratio is small enough.
This paper follows a recent bibliographic line which tackles this issue with field-theoretical methods.
Previous analysis focused on equilibrium Boltzmann distributions for some effective field representing the degrees of freedom of data.
It was possible to establish a relation between signal detection and $`\mathbb{Z}_2`$-symmetry breaking.
In this paper, we consider a stochastic field framework inspiring by the so-called "Model A", and show that the ability to reach or not an equilibrium state is correlated with the shape of the dataset.
In particular, studying the renormalization group of the model, we show that the weak ergodicity prescription is always broken for signals small enough, when the data distribution is close to the Marchenko-Pastur (MP) law.
This, in particular, enables the definition of a detection threshold in the regime where the signal-to-noise ratio is small enough.

## Installation

You can install all dependencies using the `requirements.txt` list:

```bash
python -m venv venv/
source activate venv/bin/activate
pip install -r requirements.txt
```

Notice that the code depends crucially on [`py-pde`](https://py-pde.readthedocs.io/en/latest/).

At the time of writing, `python==3.10.12` was used for the development.

## Documentation

The documentation can be built using `sphinx` (`pip install sphinx sphinx_rtd_theme`):

```bash
sphinx-build -b html docs/source <build directory>
```

You will be able to open the file `<build directory>/index.html` in your browser.

## Tutorials

Though the easiest way to get started with the library is to use the script [`simulation_temp.py`](./simulation_temp.py) (to scan using a single _temperature_ parameter) or [`simulation_traj.py`](./simulation_traj.py) (to scan using a different parametrization), we include two jupyter notebooks to show the basic usage of the library. In particular, we show the case of the analytical Marchenko-Pastur distribution and the case of the empirical distribution of the eigenvalues of a random matrix.

For more information on the scripts, you can use ``python <script> --help`` to get the definition of all command line parameters. Notice that numerical results will be saved in a SQLite database (path provided by the user from command line): you should foresee a utility to explore such database for further processing.
