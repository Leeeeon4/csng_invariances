#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Click>=7.0",
    "numpy>=1.2",
    "matplotlib>=3.4",
    "torch>=1.9",
    "torchvision>=0.10",
    "requests>=2.26",
    "datajoint>=0.13",
    "neuralpredictors==0.0.1",
    "nnfabrik==0.0.1",
    "GitPython>=3.1",
    "scipy>=1.7",
    "pandas>=1.0",
    "rich>=10.0",
]

test_requirements = requirements.append("pytest>=3")

setup(
    author="Leon Michel Gorißen",
    author_email="leon.gorissen@rwth-aachen.de",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="CSNG: Finding Invariances in Sensory Coding contains source code for invariance detection in sensory coding.",
    entry_points={
        "console_scripts": [
            "csng_invariances=csng_invariances.cli:main",
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="csng_invariances",
    name="csng_invariances",
    packages=find_packages(include=["csng_invariances", "csng_invariances.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Leeeeon4/csng_invariances",
    version="0.0.1",
    zip_safe=False,
)
