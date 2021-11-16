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
    
    :param str substitutions: The column containing raw mutation data for each variant.
    :param list[str] phenotypes: The columns of observed phenotypes for each variant
    :param errors: The error columns associated with each phenotype, assumed to be variance (:math:`\sigma^2_y`)
    :type errors: list[str], optional
    :param tokenizer: The tokenizer converting raw mutations into one-hot encoded tensors
    :type tokenizer: lantern.dataset.tokenizer.Tokenizer
    """

    substitutions: str = attr.ib(default="substitutions")
    phenotypes: List[str] = attr.ib(default=["phenotype"])
    errors: Optional[List[str]] = attr.ib(default=None)
    tokenizer: Tokenizer = attr.ib(default=None, repr=False)

    @errors.validator
    def _errors_correct_length(self, attribute, value):
        """Check for correct length between errors and phenotypes
        """
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
            n = torch.zeros(N, len(self.errors))

        for i in range(N):

            X[i, :] = self.tokenizer.tokenize(substitutions.iloc[i])
            y[i, :] = torch.from_numpy(phenotypes.iloc[i, :].values)

            if errors is not None:
                n[i, :] = torch.from_numpy(errors.iloc[i, :].values)

        tensors = [X, y]
        if errors is not None:
            tensors.append(n)

        # tensor dataset construction
        super(_Base, self).__init__(*tensors)

    @property
    def D(self):
        """The number of dimensions of the phenotype
        """
        return len(self.phenotypes)

    @property
    def p(self):
        """The number of mutations in the dataset
        """
        return self.tokenizer.p

    def to(self, device):
        """Send to device
        """
        self.tensors = [t.to(device) for t in self.tensors]

    def meanEffects(self):
        """The mean effects of each mutation against each phenotype, returned as a (p x D) tensor
        """

        X, y = self[: len(self)][:2]
        sol, _ = torch.lstsq(y, X)

        return sol[: self.p, :]


@attr.s()
class _DataframeDataset:
    """A direct from dataframe base class, used just to enforce argument order for attrs.
    """

    df: pd.DataFrame = attr.ib(repr=False)


@attr.s()
class Dataset(_DataframeDataset, _Base):
    """The runtime option for datasets, taking a dataframe as the first argument.
    
    """

    @classmethod
    def from_sequences(
        cls,
        df,
        wildtype: str,
        sequence_column: str = "sequence",
        substitutions="substitutions",
        *args,
        **kwargs,
    ):
        """Build a Dataframe dataset using full sequences, converting to a compressed substitution string."""
        df[substitutions] = df[sequence_column].apply(
            lambda x: ":".join(
                [
                    f"{ww}{i+1}{vv}"
                    for i, (ww, vv) in enumerate(zip(list(wildtype), list(x)))
                    if ww != vv
                ]
            )
        )

        return cls(df, *args, **kwargs)


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
