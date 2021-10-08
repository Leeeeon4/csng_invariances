#!/usr/bin/env python

"""Tests for `csng_invariances` package."""

import pytest
import numpy as np
import torch

from click.testing import CliRunner

# from csng_invariances import csng_invariances
from csng_invariances import cli


np_images = np.random.random(size=(10, 1, 31, 31))
np_responses = np.random.random(size=(10, 10))

torch_images = torch.randn((10, 1, 31, 31))
torch_responses = torch.randn((10, 10))


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


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.main)
    assert result.exit_code == 0
    assert "csng_invariances.cli.main" in result.output
    help_result = runner.invoke(cli.main, ["--help"])
    assert help_result.exit_code == 0
    assert "--help  Show this message and exit." in help_result.output
