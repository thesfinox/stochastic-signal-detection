Stochastic Signal Detection (SSD)
=================================

    | `Harold Erbin <mailto:harold.erbin@cea.fr>`_, `Riccardo Finotello <mailto:riccardo.finotello@cea.fr>`_, `Bio Wahabou Kpera <mailto:wahaboukpera@gmail.com>`_, `Vincent Lahoche <mailto:vincent.lahoche@cea.fr>`_, `Dine Ousmane Samary <mailto:dine.ousmanesamary@cipma.uac.bj>`_
    | |arxiv|_ |github|_

.. |arxiv| image:: https://img.shields.io/badge/arxiv-2023.07499-red
   :alt: arXiv.org
.. _arxiv: https://arxiv.org/abs/2310.07499

.. |github| image:: https://img.shields.io/badge/github-stochastic--signal--detection-blue?logo=github
   :alt: github.com
.. _github: https://github.com/thesfinox/stochastic-signal-detection


Signal detection is one of the main challenges of data science.
As it often happens in data analysis, the signal in the data may be corrupted by noise.
There is a wide range of techniques aimed at extracting the relevant degrees of freedom from data.
However, some problems remain difficult.
It is notably the case of signal detection in almost continuous spectra when the signal-to-noise ratio is small enough.
This paper follows a recent bibliographic line which tackles this issue with field-theoretical methods.
Previous analysis focused on equilibrium Boltzmann distributions for some effective field representing the degrees of freedom of data.
It was possible to establish a relation between signal detection and :math:`\mathbb{Z}_2`-symmetry breaking.
In this paper, we consider a stochastic field framework inspiring by the so-called "Model A", and show that the ability to reach or not an equilibrium state is correlated with the shape of the dataset.
In particular, studying the renormalization group of the model, we show that the weak ergodicity prescription is always broken for signals small enough, when the data distribution is close to the Marchenko-Pastur (MP) law.
This, in particular, enables the definition of a detection threshold in the regime where the signal-to-noise ratio is small enough.
