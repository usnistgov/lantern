Tutorial
########

This tutorial gives an overview of how to use the LANTERN library for
analyzing genotype-phenotype landscape (GPL) datasets.

Genotype phenotype dataset
==========================

LANTERN provides a pytorch :class:`dataset
<lantern.dataset.dataset.\_Base>` type for GPL datasets. Currently,
the implementation assumes that the raw GPL data will come in the form
of a :class:`pandas dataframe<pandas.DataFrame>`. At minimum, this
dataframe should have a column encoding the raw mutational data and
one or more phenotypes. It is also possible to include error
measurements for each phenotype, currently assumed to by observation variance.

>>> df = pd.read_csv("my-gpl.csv")
>>> df.head()
fdsa


Tokenizer
---------

Each dataset relies on a :class:`~lantern.dataset.tokenizer.Tokenizer`
to convert raw mutational data into a one-hot encoded tensor for each
variant. If this is not provided when creating the
:class:`dataset<lantern.dataset.dataset.\_Base>`, it is automatically
constructed from the available data. Usually this is what you want,
because LANTERN will automatically determine all of the mutations
present in the provided dataset. But, if additional mutations will be
incorporated later (e.g. if all possible mutations to be considered
are not in the provided dataset), then this tokenizer should be
provided directly. See the
:meth:`~lantern.dataset.tokenizer.Tokenizer.fromVariants` builder
method for more details.
