#!/usr/bin/env python

"""Tests for `csng_invariances` package."""

import pytest
from click.testing import CliRunner
import numpy as np
import torch

from csng_invariances import csng_invariances
from csng_invariances import cli
from csng_invariances.utility.data_helpers import get_test_dataset
from csng_invariances.utility.lin_filter import *


train_images, train_responses, val_images, val_responses = get_test_dataset()
reg_factors = [1 * 10 ** x for x in np.linspace(0, 2, 10)]
GlobFilTrain = GlobalRegularizationFilter(train_images, train_responses)
GlobFilVal = GlobalRegularizationFilter(val_images, val_responses)


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string


#   TODO Test everything
def test_global_linear_filters(GlobFilTrain):
    # train function
    glob_fil = GlobFilTrain.train()
    assert glob_fil.shape == (
        14,
        1,
        12,
        13,
    ), f"Shape of filter is not 4D tensor as expected. Shape is {glob_fil.shape}"

    # predict function
    glob_preds, glob_corr = GlobFilTrain.predict(fil=glob_fil)
    assert (
        glob_preds.shape == train_responses.numpy().shape
    ), f"Shape of prediction does not match shape of training response data. Shape is {glob_preds.shape} it should be {train_responses.numpy().shape}"
    # TODO correct type check of glob_corr
    # assert type(glob_corr) == (
    #     float or np.float64
    # ), f"Average correlation is no float. Type is {type(glob_corr)}"

    # evaluate function
    glob_dict = GlobFilTrain.evaluate()
    assert (
        len(glob_dict) == train_responses.shape[1]
    ), f"Length of neuron: single neuron correlation is not as expected. Lenght is {len(glob_dict)} while it should be {train_responses.shape[1]}."


def test_global_hyperparametersearch(GlobFilTrain, GlobFilVal, reg_factors):
    GlobHypSearch = GlobalHyperparametersearch(
        GlobFilTrain, GlobFilVal, reg_factors, False
    )
    GlobHypSearch.conduct_search()
    GlobHypSearch.compute_best_parameter()
    glob_param = GlobHypSearch.get_parameters()
    assert (
        len(glob_param) == 1
    ), f"Did not get a single parameter. Recieved a list of length {len(glob_param)}"


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "csng_invariances.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output


if __name__ == "__main__":
    test_global_linear_filters(GlobFilTrain)
