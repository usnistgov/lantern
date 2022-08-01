#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


setup(
    name="lantern",
    version="0.1.2a",
    license="NIST",
    description="Genotype-phenotype landscape interpretable nonparametric modeling",
    author="Dr. Peter Tonner",
    author_email="peter.tonner@nist.gov",
    url="https://gitlab.nist.gov/ptonner/lantern",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    project_urls={},
    keywords=["computational biology", "machine learning"],
    install_requires=[
        "gpytorch",
        "pandas",
        "numpy",
        "matplotlib",
        "torch",
        "attrs>=21.1.0",
    ],
    tests_requires=["pytest", "pytest-cov"],
    extras_require={},
    # entry_points={"console_scripts": ["lantern=lantern.cli:main"]},
)
