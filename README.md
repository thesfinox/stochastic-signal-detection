# Stochastic Signal Detection (SSD)

[Harold Erbin](mailto:harold.erbin@cea.fr), [Riccardo Finotello](mailto:riccardo.finotello@cea.fr), [Bio Wahabou Kpera](mailto:wahaboukpera@gmail.com), [Vincent Lahoche](mailto:vincent.lahoche@cea.fr), [Dine Ousmane Samary](mailto:dine.ousmanesamary@cipma.uac.bj)

![arXiv](https://img.shields.io/badge/arxiv-2023.XXXXX-red)
![github](https://img.shields.io/badge/github-stochastic--signal--detection-blue?logo=github)

Signal detection is one of the main challenges of data science.
As it often happens when tiding datasets, the signal in the data may be corrupted by noise.
There is a wide range of data analysis methods aimed at extracting the relevant degrees of freedom from a data.
However, some problems remain difficult.
It is notably the case of signal detection in almost continuous spectra.
This paper follows a recent bibliographic line, aiming to tackle this issue with field theoretical methods.
Previous analysis focused on equilibrium Boltzmann distributions for some effective field representing degrees of freedom of data.
It was possible to establish a relation between signal detection and $`\mathbb{Z}_2`$-symmetry breaking.
In this paper, we introduce a stochastic field formalism to address the same issue.
It follows from a reflection on the role of the statistical properties of fields in the definition of a natural time, from which the ability to reach or not an equilibrium state is in relation with the shape of the dataset.
In particular, studying the renormalization group of the model, we show that weak ergodicity prescription is (almost) always broken for signal small enough when datasets are close to the Marchenko-Pastur law.

## Installation

You can install all dependencies using the `requirements.txt` list:

```bash
python -m venv venv/
source activate venv/bin/activate
pip install -r requirements.txt
```

At the time of writing, `python==3.10.12` was used for the development.

## Tutorials

Though the easiest way to get started with the library is to use the script [`simulation.py`](./simulation.py), we include two jupyter notebooks to show the basic usage of the library. In particular, we show the case of the analytical Marchenko-Pastur distribution and the case of the empirical distribution of the eigenvalues of a random matrix.

For more information on the script [`simulation.py`](./simulation.py), you can use ``python simulation.py --help`` to get the definition of all command line parameters.
