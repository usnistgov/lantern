LANTERN: an interpretable genotype-phenotype landscape model
============================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   examples/index
   api

What is LANTERN?
================
LANTERN is a tool for learning interpretable models of
genotype-phenotype landscape (GPL) data.

Installation
============

LANTERN currently must be installed from source. It is recommended to
install in a virtual environment (e.g. venv or conda)::

  python -m pip install git+https://github.com/ptonner/lantern.git@XXX

Quickstart
==========
LANTERN provides a straightforward interface for training models:
::

   import pandas as pd
   from lantern.dataset import Dataset
   from lantern.model import Model
   from lantern.model.basis import VariationalBasis
   from lantern.model.surface import Phenotype

   # create a dataframe containing GPL data
   df = pd.DataFrame(
       substitutions=["", "+a", "+b", "+a:+b"],
       phenotype=[0.0, 1.0, 1.0, 0.8],
   )

   # convert the data to a LANTERN dataset
   ds = Dataset(df)
   
   # build a LANTERN model based on the dataset, using an upper-bound
   # of 8 latent dimensions
   model = Model(
       VariationalBasis.fromDataset(ds, 8),
       Phenotype.fromDataset(ds, 8)
   )

   loss = model.loss(N=len(ds))
   X, y = ds[:len(ds)]

   optimizer = Adam(loss.parameters(), lr=0.01)
   for i in range(100):
       optimizer.zero_grad()
       lss = loss(X, y)
       total = sum(lss.values())
       total.backwards()
       optimizer.step()

For a more thorough introduction, see the :ref:`tutorial`.

Citation
========
LANTERN can be cited as: <insert biorxiv link>

The workflow used for generating the results of the manuscript is
available at `github.com/ptonner/lantern/manuscript`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
