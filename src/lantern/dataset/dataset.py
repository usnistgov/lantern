"""Custom pytorch dataset for genotype-phenotype landscapes."""

from typing import List, Optional

from torch.utils.data import TensorDataset
import torch
import attr
import pandas as pd
import numpy as np

from lantern.dataset.tokenizer import Tokenizer


@attr.s()
class _Base(TensorDataset):
    """Base genotype-phenotype dataset class, shuttling a pandas dataframe
    to a TensorDataset.

    """

    substitutions: str = attr.ib(default="substitutions")
    phenotypes: List[str] = attr.ib(default=["phenotype"])
    errors: Optional[List[str]] = attr.ib(default=None)
    tokenizer: Tokenizer = attr.ib(default=None, repr=False)

    @errors.validator
    def _errors_correct_length(self, attribute, value):
        if value is not None and len(value) != len(self.phenotypes):
            raise ValueError(
                f"Number of error columns ({len(value)}) does not match phenotype columns ({len(self.phenotypes)})"
            )

    def __attrs_post_init__(self):

        # Extract components from the dataframe. These will be
        # used to construct the needed tensors.
        substitutions = self.df[self.substitutions].replace(np.nan, "")
        phenotypes = self.df[self.phenotypes]
        if self.errors is not None:
            errors = self.df[self.errors]
        else:
            errors = None

        # We need a way to convert substitutions into tokens. If this
        # hasn't been provided, then create it from the avaialble
        # data. If all tokens are not part of the provide dataframe,
        # the tokenizer should be built with the necessary additional
        # information.
        if self.tokenizer is None:
            self.tokenizer = Tokenizer.fromVariants(substitutions)

        # build tensors
        N = len(self.df)

        X = torch.zeros(N, self.tokenizer.p)
        y = torch.zeros(N, len(self.phenotypes))
        if errors is not None:
            n = torch.zeros(N, self.len(self.errors))

        for n in range(N):

            X[n, :] = self.tokenizer.tokenize(substitutions.iloc[n])
            y[n, :] = torch.from_numpy(phenotypes.iloc[n, :].values)

            if errors is not None:
                n[n, :] = torch.from_numpy(errors.iloc[n, :].values)

        tensors = [X, y]
        if errors is not None:
            tensors.append(n)

        # tensor dataset construction
        super(Dataset, self).__init__(*tensors)

    @property
    def D(self):
        return len(self.phenotypes)

    @property
    def p(self):
        return self.tokenizer.p


@attr.s()
class _DataframeDataset:
    """A direct from dataframe base class, used just to enforce argument order for attrs.
    """

    df: pd.DataFrame = attr.ib()


@attr.s()
class Dataset(_DataframeDataset, _Base):
    """The runtime option for datasets, taking a dataframe as the first argument.
    """

    pass


@attr.s()
class _Csv:
    """Dataset from csv target, used just to enforce argument order for attrs.
    """

    pth: str = attr.ib()


@attr.s()
class CsvDataset(_Csv, _Base):
    def __attrs_post_init__(self):
        self.df = pd.read_csv(self.pth)
        super(CsvDataset, self).__attrs_post_init__()
