===========================================
CSNG: Finding Invariances in Sensory Coding
===========================================

CSNG: Finding Invariances in Sensory Coding contains source code for invariance detection in sensory coding.

.. image:: https://img.shields.io/pypi/l/csng_invariances.svg
   :target: https://pypi.python.org/pypi/csng_invariances/

.. image:: https://img.shields.io/pypi/pyversions/csng_invariances.svg
   :target: https://pypi.python.org/pypi/csng_invariances/

.. image:: https://readthedocs.org/projects/csng_invariances/badge/?version=latest
   :target: https://csng_invariances.readthedocs.io/?badge=latest



* Free software: MIT license
* Repository: https://github.com/Leeeeon4/csng_invariances
* Documentation: https://csng-invariances.readthedocs.io


Features
--------

* `PyTorch`_ implementation of Kovács 2021: Finding invariances in sensory coding.

.. _`PyTorch`: https://pytorch.org

* Access to custom Dataset based on data presented in `Antolik et al. 2016`_.

.. _`Antolik et al. 2016`: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004927#abstract0

* Several possible estimations for linear receptive fields, i.e. filter computations for encoding visual stimuli to neural responses linearly. Computations behave analogously to `Spike-triggered Average`_ computations.

.. _`Spike-triggered Average`: https://en.wikipedia.org/wiki/Spike-triggered_average

* Estimation for non-linear receptive fields as presented in `Lurz et al. 2020`_.

.. _`Lurz et al. 2020`: https://openreview.net/forum?id=Tp7kI90Htd

* `Weights and Biases`_ integration. 

.. _`Weights and Biases`: https://wandb.ai/csng-cuni/


Credits
-------

This work is based on Kovács 2021: Finding invariances in sensory coding.

The directory structure is adapted from the `datadriven/cookiecutter-data-science`_ project template.

.. _`datadriven/cookiecutter-data-science`: https://github.com/drivendata/cookiecutter-data-science

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
