LANTERN: an interpretable genotype-phenotype landscape model
============================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorial
   api

Installation
============

LANTERN can be installed from pip::

  pip install lantern-gpl

Quickstart
==========
::

   import pandas as pd
   from lantern.dataset import Dataset
   from lantern.model import Model
   from lantern.model.basis import VariationalBasis
   from lantern.model.surface import Phenotype

   df = pd.DataFrame(
       substitutions=["", "+a", "+b", "+a:+b"],
       phenotype=[0.0, 1.0, 1.0, 0.8],
   )

   ds = Dataset(df)
   
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

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
