import attr
import numpy as np
import pandas as pd
import torch

from lantern.dataset.dataset import Dataset
from lantern.model.model import Model


@attr.s()
class Experiment:

    dataset: Dataset = attr.ib()
    model: Model = attr.ib()
    
    
    def prediction_table(self, mutations_list=None, max_variants=None):
    
        dataset = self.dataset
        model = self.model
        
        # Start with list of string representations of the mutations in each variant to predict
        #     default is to use the variants in dataset
        if mutations_list is None:
            sub_col = dataset.substitutions
            mutations_list = list(dataset.df[sub_col].replace(np.nan, ""))
            
            if max_variants is not None:
                mutations_list = mutations_list[:max_variants]
        
        elif type(mutations_list) is not list:
            mutations_list = list(mutations_list)
                        
        # Convert from list of mutation strings to one-hot encoding
        tok = dataset.tokenizer
        X = tok.tokenize(*mutations_list)
        
        # The tokenize() method returns a 1D tensor if len(mutations_list)==1
        #     but here, we always want a 2D tensor
        if len(X.shape) == 1:
            X = X.unsqueeze(0)
        
        # Get Z coordinates (latent space) for each variant
        Z = model.basis(X)
        
        # Get predicted mean phenotype and variance as a function of Z coordinates
        f = model.surface(Z)
        with torch.no_grad():
            fmu = f.mean.numpy()
            fvar = f.variance.numpy()
            Z_arr = Z.numpy()
        
        # Make the datafream to return
        df_return = pd.DataFrame({'substitutions':mutations_list})
        
        # Add predicted phenotype columns
        df_columns = dataset.phenotypes
        df_columns = [x.replace('-norm', '') for x in df_columns]
        for c, f_c in zip(df_columns, fmu.transpose()):
            df_return[c] = f_c
            
        # Add columns for Z coordinates
        for i, z in enumerate(Z_arr.transpose()):
            df_return[f'z_{i+1}'] = z
        
        return df_return
