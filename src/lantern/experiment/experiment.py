import attr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from lantern.dataset.dataset import Dataset
from lantern.model.model import Model



@attr.s()
class Experiment:

    dataset: Dataset = attr.ib()
    model: Model = attr.ib()
    
    
    def prediction_table(self, 
                         mutations_list=None, 
                         max_variants=None, 
                         uncertainty_samples=50, 
                         batch_size=32, 
                         uncertainty=False,
                         verbose=False):
    
        dataset = self.dataset
        tok = dataset.tokenizer
        model = self.model
        phenotypes = dataset.phenotypes
        errors = dataset.errors
        
        # Start with list of string representations of the mutations in each variant to predict
        #     default is to use the variants in dataset
        if mutations_list is None:
            sub_col = dataset.substitutions
            # in case the substitutions entry for the WT variants is marked as np.nan (it should be an empty string)
            mutations_list = list(dataset.df[sub_col].replace(np.nan, ""))
            
            if max_variants is not None:
                mutations_list = mutations_list[:max_variants]
            
        else:
            df = pd.DataFrame({'substitutions':mutations_list})
            for c in list(phenotypes) + list(errors):
                df[c] = 0
            dataset = Dataset(df, phenotypes=phenotypes, errors=errors)
        
        if type(mutations_list) is not list:
            mutations_list = list(mutations_list)
        
        if batch_size == 0:     
            # Convert from list of mutation strings to one-hot encoding
            X = tok.tokenize(*mutations_list)
            
            # The tokenize() method returns a 1D tensor if len(mutations_list)==1
            #     but here, we always want a 2D tensor
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            
            # Get Z coordinates (latent space) for each variant
            Z = model.basis(X)
            
            # Get predicted mean phenotype and variance as a function of Z coordinates
            # 
            # The next line is equivalent to: f = model(X): f = model(X) is equivalent to f = model.forward(X), which is equivalent to f = model.surface(model.basis(X)).
            f = model.surface(Z) 
            with torch.no_grad():
                Y = f.mean.numpy() # Predicted phenotype values
                Z = Z.numpy() # latent space coordinates
        else:
            Y = [] # Predicted phenotype values
            Z = [] # latent space coordinates
            if uncertainty:
                Yerr = []
                Zerr = []
                        
            mutations_list_list = [mutations_list[pos:pos + batch_size] for pos in range(0, len(mutations_list), batch_size)]
            
            j = 0
            if verbose: print(f'Number of batches: {len(mutations_list_list)}')
            for mut_list in mutations_list_list:
                if verbose: print(f'Batch: {j}')
                # _x is the one-hot encoding for the batch.
                _x = tok.tokenize(*mut_list)
                if len(_x.shape) == 1:
                    _x = _x.unsqueeze(0)
                
                _z = model.basis(_x)
                if isinstance(_z, tuple):
                    _z = _z[0]
                
                _f = model.surface(_z)
                
                with torch.no_grad():
                    _y = _f.mean.numpy()
                    _z = _z.numpy()
                    
                    if uncertainty:
                        f_tmp = torch.zeros(uncertainty_samples, *_y.shape)
                        z_tmp = torch.zeros(uncertainty_samples, *_z.shape)
                        model.train()
                        for n in range(uncertainty_samples):
                            z_samp = model.basis(_x)
                            z_tmp[n, :, :] = z_samp
                            f_samp = model(_x)
                            samp = f_samp.sample()
                            if samp.ndim == 1:
                                samp = samp[:, None]

                            f_tmp[n, :, :] = samp

                        _yerr = f_tmp.std(axis=0)
                        _zerr = z_tmp.std(axis=0)

                        model.eval()
                
                Y += list(_y)
                Z += list(_z)
                if uncertainty:
                    Yerr += list(_yerr)
                    Zerr += list(_zerr)
                
                j += 1
            Y = np.array(Y)
            Z = np.array(Z)
            if uncertainty:
                Yerr = np.array(Yerr)
                Zerr = np.array(Zerr)
            
        # Fix ordering of Z dimensions from most to least important
        Z = Z[:, model.basis.order]
        if uncertainty:
            Zerr = Zerr[:, model.basis.order]
        
        
        # Make the datafream to return
        df_return = pd.DataFrame({'substitutions':mutations_list})
        
        # Add predicted phenotype columns
        df_columns = dataset.phenotypes
        df_columns = [x.replace('-norm', '') for x in df_columns]
        for c, y in zip(df_columns, Y.transpose()):
            df_return[c] = y
        if uncertainty:
            for c, yerr in zip(df_columns, Yerr.transpose()):
                df_return[f'{c}_err'] = yerr
            
        # Add columns for Z coordinates
        for i, z in enumerate(Z.transpose()):
            df_return[f'z_{i+1}'] = z
        if uncertainty:
            for i, zerr in enumerate(Zerr.transpose()):
                df_return[f'z_{i+1}_err'] = zerr
        
        return df_return
    
    
    def dim_variance_plot(self, ax=None, include_total=False, figsize=[4, 4], **kwargs):
        # Plots the variance for each of the dimensions in the latent space - used to identify which dimonesions are "importaant"
    
        model = self.model

        mean = 1 / model.basis.qalpha(detach=True).mean[model.basis.order]
        z_dims = [n + 1 for n in range(len(mean))]
        
        if ax is None:
            plt.rcParams["figure.figsize"] = figsize
            fig, ax = plt.subplots()
        
        ax_twin = ax.twiny()
        
        ax_twin.plot(z_dims, mean, "-o")

        ax_twin.set_xlabel("Z dimension")
        ax_twin.set_xticks(z_dims)
        ax_twin.set_ylabel("variance")

        mn = min(mean.min(), 1e-4)
        mx = mean.max()
        z = torch.logspace(np.log10(mn), np.log10(mx), 100)
        ax.plot(
            invgammalogpdf(z, torch.tensor(0.001), torch.tensor(0.001)).exp().numpy(),
            z.numpy(),
            c="k",
            zorder=0,
        )
        ax.set_xlabel("prior probability")

        ax.set_yscale('log')


def invgammalogpdf(x, alpha, beta):
    return alpha * beta.log() - torch.lgamma(alpha) + (-alpha - 1) * x.log() - beta / x

