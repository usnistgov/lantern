Tutorial
########

This tutorial gives an overview of how to use the LANTERN library for
analyzing genotype-phenotype landscape (GPL) datasets.

Genotype phenotype dataset
==========================

LANTERN provides a :class:`pytorch dataset
type<lantern.dataset.\_Base>` for GPL datasets. Currently, the
implementation assumes that the raw GPL data will come in the form of
a :class:`pandas dataframe<pandas.DataFrame>`. At minimum, this
dataframe should have a column encoding the raw mutational data
(substitutions) and one or more phenotypes (:py:attr:`lantern.dataset.\_Base.phenotypes`) 

Tokenizer
---------

Each dataset converts raw mutational data
