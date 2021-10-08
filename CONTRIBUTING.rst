.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/Leeeeon4/csng_invariances/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

CSNG: Finding Invariances in Sensory Coding could always use more documentation, whether as part of the
official CSNG: Finding Invariances in Sensory Coding docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/Leeeeon4/csng_invariances/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `csng_invariances` for local development.

1. Fork the `csng_invariances` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/csng_invariances.git

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv csng_invariances
    $ cd csng_invariances/
    $ python setup.py develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the
   tests, including testing other Python versions with tox::

    $ flake8 csng_invariances tests
    $ python setup.py test or pytest
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.7, 3.8 and 3.9, and for PyPy. Check
   https://travis-ci.com/Leeeeon4/csng_invariances/pull_requests
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ pytest tests.test_csng_invariances

Repository Structure
--------------------

The GitHub Repository is structured as described bellow:

::

    Repository
    ├── LICENSE
    ├── Makefile                Makefile with commands like `make docs`.
    ├── README.rst              The top-level README for developers 
                                using this project.
    ├── csng_invariances        Package source code. See Documentation for more
                                information.
    ├── data
        ├── external            Data from third party sources.
        ├── interim             Intermediate data that has been 
                                transformed.
        ├── processed           The final, canonical data sets for 
                                modeling.
        └── raw                 The original, immutable data dump.
    ├── docs                    A default Sphinx project; see 
                                sphinx-doc.org for details
    ├── models                  Trained and serialized models, model 
                                predictions, or model summaries
    ├── notebooks               Jupyter notebooks. Naming convention is
                                a number (for ordering), the creator's
                                initials, and a short `-` delimited
                                description, e.g.
                                `1.0-jqp-initial-data-exploration`.
    ├── references              Data dictionaries, manuals, and all 
                                other explanatory materials.
    ├── reports                 Generated analysis as HTML, PDF, LaTeX,
                                etc.
        └── figures             Generated graphics and figures to be 
                                used in reporting
    ├── requirements_dev.txt    The requirements file for reproducing 
                                the analysis environment, e.g. 
                                generated with 
                                `pip freeze > requirements_dev.txt`
    ├── setup.py                makes project pip installable 
                                (pip install -e .) so `csng_invariances`
                                can be imported.
    ├── tests                   Source code for testing of package.
    └── tox.ini                 tox file with settings for running tox;
                                see tox.readthedocs.io

Makefile
--------

The Makefile provides useful operations:

* Help: Show help (list all make commands in console)::

    $ make help

* Clean: remove all build, test, coverage and Python artifacts::

    $ make clean

* Clean-build: remove build artifacts::

    $ make clean-build

* Clean-pyc: remove Python file artifacts::

    $ make clean-pyc

* Clean-test: remove test and coverage artifacts::

    $ make clean-test

* Lint: check style with flake8::

    $ make lint

* Test: run tests quickly with the default Python::

    $ make test

* Test-all: run tests on every Python version with tox::

    $ make test-all

* Coverage: check code coverage quickly with the default Python::

    $ make coverage

* Documentation: generate Sphinx HTML documentation, including API docs::

    $ make docs

* Servedocs: compile the docs watching for changes::

    $ make servedocs

* Release: package and upload a release::

    $ make release

* Distribution: builds source and wheel package::

    $ make dist

* Installation: install the package to the active Python's site-packages::
    
    $ make install

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in HISTORY.rst).
Then run::

$ bump2version patch # possible: major / minor / patch
$ git push
$ git push --tags

Travis will then deploy to PyPI if tests pass.
