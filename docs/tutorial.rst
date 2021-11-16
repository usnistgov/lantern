.. _tutorial:

Tutorial
########

This tutorial gives an overview of how to use the LANTERN library for
analyzing genotype-phenotype landscape (GPL) datasets.

Genotype phenotype dataset
==========================

GPL Dataset
-----------

LANTERN provides a pytorch :class:`Dataset
<lantern.dataset.dataset.\_Base>` type for GPL datasets. Currently,
the implementation assumes that the raw GPL data will come in the form
of a :class:`pandas DataFrame<pandas.DataFrame>`. At minimum, the
DataFrame should have:

1. A `substitutions` column representing the string encoded mutational
   data of each variant. For example, the string `"+a:+b"` represents
   a variant with the mutations `"+a"` and `"+b"`. To control how
   mutation strings are processed, see :ref:`tokenizer`.
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

To create the actual :class:`Dataset<lantern.dataset.dataset.\_Base>`
object, provide the DataFrame

>>>  from lantern.dataset import Dataset
>>>  ds = Dataset(df)
>>>  ds
Dataset(substitutions='substitutions', phenotypes=['phenotype'], errors=None)
>>>  len(ds)
32
>>>  ds[0] # get the first element (a tuple of x_0, y_0)
(tensor([0., 0., 0., 0., 0.]), tensor([0.7598]))

The genotype of each variant is represented as a one-hot encoded
vector, where `0` or `1` indicate absence or presence of a mutation,
respectively. To see what each element of :math:`x_i` corresponds to,
you can look at the dataset
:class:`~lantern.dataset.tokenizer.Tokenizer`.

>>>  ds.tokenizer.tokens
['+a', '+b', '+c', '+d', '+e']

You can also easily intraconvert between one-hot encoded and mutation
strings:

>>>  ds.tokenizer.tokenize("+a:+c")
tensor([1., 0., 1., 0., 0.])
>>>  ds.tokenizer.detokenize(ds[14][0])
'+c:+e'


An alternative data format is supported, comparing the full sequence
for each variant against the wild-type (see :meth:`~lantern.dataset.Dataset.from_sequences`).

>>> df = pd.DataFrame({"phenotype": [0]*3, "sequence": ["AAA", "ATC", "CGA"]})
>>> ds = lantern.dataset.Dataset.from_sequences(df, wildtype="AAA", sequence_col="sequence")
>>> ds.tokenizer.tokens
['A1C', 'A2G', 'A2T', 'A3C']




Building a LANTERN model
========================

A LANTERN model is composed of two key elements:

1. A :class:`~lantern.model.basis.Basis` that performs a linear
   transformation of a one-hot encoded mutational vector to a
   low-dimensional latent mutational effect space
2. A :class:`~lantern.model.surface.Surface` that models the
   non-linear relationship between the latent mutational effect space
   and the observed phenotypes

Mathematically, this two-step operation can be seen as

.. math:: z_i = W x_i
          :label: eq_basis
.. math:: y_i = f(z_i)
          :label: eq_surface

For a basis :math:`W` and non-linear surface :math:`f` to be learned
from the data. LANTERN provides a python interface for this learning
problem.

Currently, there is a single interface to both the `Basis` and
`Surface` elements. There are built-in factory methods for both
objects using the `Dataset` object we have already created:

>>>  basis = VariationalBasis.fromDataset(ds, K=8, meanEffectsInit=True)
>>>  surface = Phenotype.fromDataset(ds, K=8)

To speed-up inference, we initialize the first dimension of the basis
to the mean effects of each mutation (`meanEffectsInit=True`).

The argument `K` describes the *maximum* number of possible latent
dimensions to be discovered in the data. In general, `K` should be
large enough to ensure that all relevant dimensions are learned by the
model. For reference, five latent dimensions were sufficient for a
dataset with over 100,000 observations. Our example dataset with only
32 examples was simulated with a single latent dimension. But, LANTERN
should learn this from the data rather than needing a "hard-coded"
value. In general, `K=8` is a reasonable default value but if all
latent dimensions are active in your learned model then consider
increasing this value.

After creating our `Basis` and `Surface`, we can now create a unified
:class:`~lantern.model.Model`

>>>  model = Model(basis, surface)

Training a LANTERN model
========================

In order to learn the components of the LANTERN model, we have to
build a :class:`~lantern.loss.Loss` for training. This is made
straightforward

>>>  loss = model.loss(N=len(ds))

We have to provide the size of the dataset (`N=len(ds)`) to ensure the
proper balance between model complexity and evidence provided by the
data.

This loss can then be used to optimize the model. We provide a
standard training procedure here:

>>>  from torch.optim import Adam
>>>  optimizer = Adam(loss.parameters(), lr=0.01)
>>>  for i in range(100):
>>>      optimizer.zero_grad()
>>>      yhat = model(X)
>>>      lss = loss(yhat, y)
>>>      total = sum(lss.values())
>>>      total.backward()
>>>      optimizer.step()

The results should like


.. plot:: plots/training.py

Downstream Analysis
===================

Examples of downstream analysis of trained models can be seen in the :ref:`Examples<examples>`.



