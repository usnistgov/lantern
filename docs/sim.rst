.. _simulate:

Simulating a GPL dataset
========================

To generate a simulated GPL dataset, we do:

.. plot:: plots/sim-data.py
          :include-source:

We can build a dataset from this simulation by::
  
  df = pd.DataFrame(
      {
          "substitutions": [
              ":".join(
                  [
                      "+{}".format(string.ascii_lowercase[i])
                      for i in np.where(X[j, :].numpy())[0]
                  ]
              )
              for j in range(X.shape[0])
          ],
          "phenotype": y,
      },
  )
  df.to_csv("simulated.csv", index=False)
