.. _tutorial:

Tutorial
########

This tutorial gives an overview of how to use the LANTERN library for
analyzing genotype-phenotype landscape (GPL) datasets.

Genotype phenotype dataset
==========================

LANTERN provides a pytorch :class:`dataset
<lantern.dataset.dataset.\_Base>` type for GPL datasets. Currently,
the implementation assumes that the raw GPL data will come in the form
of a :class:`pandas DataFrame<pandas.DataFrame>`. At minimum, the
DataFrame should have:

1. A `substitutions` column representing the string encoded mutational
   data of each variant. For example, the string `"+a:+b"` represents
   a variant with the mutations `"+a"` and `"+b"`
2. One or more `phenotypes` columns measured for each variant
3. An optional matching set of `errors` columns (one for each
   phenotype). Assumed to be observed variance

An example dataset (see :ref:`simulate`):

>>> df = pd.read_csv("example.csv")
>>> df.head()
  substitutions  phenotype
0           NaN   0.759801
1            +a   1.141440
2            +b   0.159070
3            +c   0.075118
4            +d   0.807002

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
