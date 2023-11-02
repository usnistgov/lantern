import attr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from sklearn import metrics
from scipy import stats

import torch

from lantern.dataset.dataset import Dataset
from lantern.model.model import Model



@attr.s()
class Experiment:

    dataset: Dataset = attr.ib()
    model: Model = attr.ib()
    
    
    def prediction_table(self, 
                         mutations_list=None, # input list of substitutions to predict. If None, the full input data for the experiment is used.
                         max_variants=None, # Only used if mutations_list is None. The max number of input substitutions to predict
                         uncertainty_samples=50, # Number of random draws used to estimate uncertainty of predictions.
                         batch_size=500, # batch size used in calculating predictions and uncertainties. If zero, then the method runs all the predictions in a single batch.
                         uncertainty=False, # Boolean to indicate whether or not to include uncertainties in the output table. If batch_size is zero, this parameter is ignored.
                         predict_from_z=False, # If true, the method ignores the mutations_list and predicts based on an input set of latent-space (Z) vectors
                         z_input=None, # Required if predict_from_z is True. Array/list of latent-space (Z) vectors. 
                         #                   z_input.shape should be N x D, where N is the number of z-vectors to predict and D is the number of dimensions in the latent space.
                         verbose=False): # Whether or not to print extra output.
    
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
        
        if (batch_size == 0) or predict_from_z:     
            # Convert from list of mutation strings to one-hot encoding
            X = tok.tokenize(*mutations_list)
            
            # The tokenize() method returns a 1D tensor if len(mutations_list)==1
            #     but here, we always want a 2D tensor
            if len(X.shape) == 1:
                X = X.unsqueeze(0)
            
            if predict_from_z:
                with torch.no_grad():
                    Z = torch.Tensor(z_input)
                z_order = model.basis.order
                z_re_order = np.array([np.where(z_order==n)[0][0] for n in range(len(z_order))])
                Z = Z[:, z_re_order]
            else:
                # Get Z coordinates (latent space) for each variant
                Z = model.basis(X)
            
            # Get predicted mean phenotype and variance as a function of Z coordinates
            # 
            # The next line is equivalent to: f = model(X): f = model(X) is equivalent to f = model.forward(X), which is equivalent to f = model.surface(model.basis(X)).
            f = model.surface(Z) 
            with torch.no_grad():
                Y = f.mean.numpy() # Predicted phenotype values
                Yerr = np.sqrt(f.variance.numpy())
                Z = Z.numpy() # latent space coordinates
        else:
            Y = [] # Predicted phenotype values
            Z = [] # latent space coordinates
            if uncertainty:
                Yerr = []
                Yerr_0 = []
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
                    _yerr_0 = np.sqrt(_f.variance.numpy())
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
                    Yerr_0 += list(_yerr_0)
                    Zerr += list(_zerr)
                
                j += 1
            Y = np.array(Y)
            Z = np.array(Z)
            if uncertainty:
                Yerr = np.array(Yerr)
                Yerr_0 = np.array(Yerr_0)
                Zerr = np.array(Zerr)
            
        # Fix ordering of Z dimensions from most to least important
        Z = Z[:, model.basis.order]
        
        if uncertainty and (not predict_from_z) and (batch_size != 0):
            Zerr = Zerr[:, model.basis.order]
            Yerr = np.max([Yerr, Yerr_0], axis=0) # make sure the error estimate returned is at least as big as the GP error at fixed Z
        
        # Make the datafream to return
        if predict_from_z:
            df_return = pd.DataFrame({'z1':Z.transpose()[0]})
        else:
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
        if uncertainty and not predict_from_z:
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
        
    def latent_space_plot(self,
                          df_pred=None, # DataFrame with phenotypes and z-coordinates for the scatterplot. If None, the mutations_list is used to get a prediction_table().
                          df_exp=None,
                          mutations_list=None, # input list of substitutions to predict. If None, the full input data for the experiment is used.
                          z_dims=[1,2],
                          phenotype=None, #If not string, then a list of tuples defining linear mixture of phenotypes
                          phenotype_label=None,
                          mark_wt=True,
                          wt_marker='X',
                          xlim=None, ylim=None, 
                          fig_ax=None, 
                          figsize=[4, 4],
                          sort_by_err=True,
                          colorbar=True,
                          cbar_kwargs={},
                          scatter_alpha=0.8,
                          contours=True,
                          contour_grid_points=100,
                          contour_kwargs={},
                          **kwargs):
        
        dataset = self.dataset
        if df_pred is None:
            df_pred = self.prediction_table(mutations_list=mutations_list)
        else:
            df = df_pred
            
        if df_exp is None:
            df_exp = self.dataset.df
        #test_arr = [sub in mutations_list for sub in df_exp[self.dataset.substitutions]]
        #if not np.all(test_arr):
        #    raise ValueError('All mutations in mutations_list are not in Experiment.dataset.df')
        
        if list(df_exp.substitutions) == list(df_pred.substitutions):
            df_c = df_exp
            df_z = df_pred
        else:
            print('Calculating matching rows for mutation_list')
            exp_ind_list = []
            pred_ind_list = []
            for mut in mutations_list:
                df = df_exp
                df = df[df.substitutions==mut]
                exp_ind_list.append(df.index[0])
                
                df = df_pred
                df = df[df.substitutions==mut]
                pred_ind_list.append(df.index[0])
                
            df_c = df_exp.loc[exp_ind_list]
            df_z = df_pred.loc[pred_ind_list]
        
        if phenotype is None:
            phenotype = dataset.phenotypes[0].replace('-norm', '')

        if fig_ax is None:
            plt.rcParams["figure.figsize"] = figsize
            fig, ax = plt.subplots()
        else:
            fig, ax = fig_ax
        
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
            
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis'
        if 's' not in kwargs:
            kwargs['s'] = 9
            
        x = df_z[f'z_{z_dims[0]}']
        y = df_z[f'z_{z_dims[1]}']
        if type(phenotype) is str:
            c = df_c[phenotype]
            cerr = np.sqrt(df_c[f'{phenotype}_var'])
        else:
            c = phenotype[0][0]*df_c[phenotype[0][1]].values
            cvar = np.abs(phenotype[0][0])*df_c[f'{phenotype[0][1]}_var'].values
            for p_tup in phenotype[1:]:
                c += p_tup[0]*df_c[p_tup[1]]
                cvar += np.abs(p_tup[0])*df_c[f'{p_tup[1]}_var']
            cerr = np.sqrt(cvar)
        
        if sort_by_err:
            df_plot = pd.DataFrame({'x':x, 'y':y, 'c':c, 'cerr':cerr})
            df_plot.sort_values(by='cerr', ascending=False, inplace=True)
            x = df_plot.x
            y = df_plot.y
            c = df_plot.c
            
        im = ax.scatter(x, y, c=c, **kwargs)
        
        ax.set_xlabel(f'$Z_{z_dims[0]}$')
        ax.set_ylabel(f'$Z_{z_dims[1]}$')
        
        if colorbar:
            ax_box = ax.get_position()
            w = ax_box.width/15
            h = ax_box.height
            x = ax_box.x1 + w
            y = ax_box.y0
            cb_ax = fig.add_axes([x, y, w, h])
            cbar = fig.colorbar(im, cax=cb_ax, **cbar_kwargs)
            cbar.solids.set(alpha=1)
            if phenotype_label is None:
                phenotype_label = phenotype
            cbar.ax.set_ylabel(phenotype_label, rotation=270, labelpad=20, size=16)
        
        if contours:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            rect = patches.Rectangle((xlim[0], ylim[0]), 
                                     xlim[1] - xlim[0], 
                                     ylim[1] - ylim[0], 
                                     edgecolor='none', 
                                     facecolor='w', 
                                     alpha=(1 - scatter_alpha))
            ax.add_patch(rect)
            
            x_points = np.linspace(*xlim, contour_grid_points)
            y_points = np.linspace(*ylim, contour_grid_points)
            x_points, y_points = np.meshgrid(x_points, y_points)
            
            # Fill out the rest of the z-vectors with zeros
            x_flat = x_points.flatten()
            y_flat = y_points.flatten()
            
            z_list = np.zeros((len(x_flat), len(self.model.basis.order)))
            
            z_list.transpose()[z_dims[0]-1] = x_flat
            z_list.transpose()[z_dims[1]-1] = y_flat
            
            df_flat = self.prediction_table(predict_from_z=True, z_input=z_list, uncertainty=False)
            
            if type(phenotype) is str:
                p_flat = df_flat[phenotype]
            else:
                p_flat = phenotype[0][0]*df_flat[phenotype[0][1]].values
                for p_tup in phenotype[1:]:
                    p_flat += p_tup[0]*df_flat[p_tup[1]]
                
            p_points = np.split(p_flat, contour_grid_points)
            
            if 'cmap' not in contour_kwargs:
                contour_kwargs['cmap'] = 'viridis'
            
            ax.contour(x_points, y_points, p_points, **contour_kwargs)
            
        if mark_wt:
            ax.plot([0], [0], wt_marker, ms=10, c='k', fillstyle='none', zorder=100, markeredgewidth=2)
        
        return fig, ax
        
        
    def latent_space_grid_plot(self,
                               df_pred=None, # DataFrame with phenotypes and z-coordinates for the scatterplot. If None, the mutations_list is used to get a prediction_table().
                               mutations_list=None, # input list of substitutions to predict. If None, the full input data for the experiment is used.
                               max_z_dim=4,
                               phenotype=None, #If not string, then a list of tuples defining linear mixture of phenotypes
                               phenotype_label=None,
                               box_size=4,
                               gridspec_space=0.35,
                               plot_lims=None,
                               **kwargs):
        
        if df_pred is None:
            df_pred = self.prediction_table(mutations_list=mutations_list)
        
        if plot_lims is None:
            plot_lims = [None]*max_z_dim
            
        if phenotype_label is None:
            phenotype_label = phenotype

        plt.rcParams["figure.figsize"] = [box_size*(max_z_dim-1), box_size*(max_z_dim-1)]

        fig, axs_grid = plt.subplots(max_z_dim-1, max_z_dim-1, gridspec_kw={'wspace':gridspec_space, 'hspace':gridspec_space})
        ax_box = axs_grid.flatten()[0].get_position()
        y = ax_box.y1 + ax_box.height/20
        x = ax_box.x0 + ax_box.width/2
        fig.suptitle(phenotype_label, size=20, x=x, y=y, va='bottom', ha='left')

        for i, axs_col in enumerate(axs_grid.transpose()):
            for j, ax in enumerate(axs_col):
                if i > j:
                    ax.set_axis_off()
                else:
                    colorbar = ax is axs_grid[0,0]
                    self.latent_space_plot(phenotype=phenotype, z_dims=[i+1, j+2], xlim=plot_lims[i], ylim=plot_lims[j+1], 
                                           df_pred=df_pred, fig_ax=(fig, ax), colorbar=colorbar, phenotype_label=phenotype_label,
                                           **kwargs);
        
        return fig, axs_grid;
    
    
    def prediction_accuracy_plot(self,
                                 phenotype,
                                 mutations_list=None,
                                 df_pred=None,
                                 df_exp=None,
                                 fig_ax=None,
                                 figsize=[4, 4],
                                 max_points=100000,
                                 alpha=0.03,
                                 colorbar=True,
                                 cbar_kwargs={},
                                 cmap='YlOrBr_r',
                                 color_by_err='experiment',
                                 sort_by_err=True):
        
        if mutations_list is None:
            sub_col = self.dataset.substitutions
            # in case the substitutions entry for the WT variants is marked as np.nan (it should be an empty string)
            mutations_list = list(self.dataset.df[sub_col].replace(np.nan, ""))
        
        if df_pred is None:
            df_pred = self.prediction_table(mutations_list=mutations_list)
        
        if df_exp is None:
            df_exp = self.dataset.df
        
        if list(df_exp.substitutions) == list(df_pred.substitutions):
            df_x = df_exp
            df_y = df_pred
        else:
            print('Calculating matching rows for mutation_list')
            exp_ind_list = []
            pred_ind_list = []
            for mut in mutations_list:
                df = df_exp
                df = df[df.substitutions==mut]
                exp_ind_list.append(df.index[0])
                
                df = df_pred
                df = df[df.substitutions==mut]
                pred_ind_list.append(df.index[0])
                
            df_x = df_exp.loc[exp_ind_list]
            df_y = df_pred.loc[pred_ind_list]
        
        if fig_ax is None:
            plt.rcParams["figure.figsize"] = figsize
            fig, ax = plt.subplots()
            
        x = df_x[phenotype].values
        xerr = np.sqrt(df_x[f'{phenotype}_var']).values
        y = df_y[phenotype].values
        if color_by_err != 'experiment':
            # Don't need yerr if only using experimental errors for colormap
            yerr = df_y[f'{phenotype}_err'].values
            err = np.sqrt(xerr**2 + yerr**2)
            
            rms = weighted_rms_residual(df_x[phenotype], df_y[phenotype], yerr=yerr, xerr=xerr)
            r2_score = metrics.r2_score(x, y, sample_weight=1/(yerr**2 + xerr**2))
        else:
            rms = weighted_rms_residual(df_x[phenotype], df_y[phenotype], yerr=None, xerr=xerr)
            r2_score = metrics.r2_score(x, y, sample_weight=1/(xerr**2))
        
        spearmanr = stats.spearmanr(x, y).statistic
        title = f'{phenotype};\nR2: {r2_score:.2f}; Spearman: {spearmanr:.2f};\nRMS resid: {rms:.2f}, {10**rms:.2f}-fold'
        fig.suptitle(title, y=0.9, size=16, va='bottom')
        
        if len(df_x) > max_points:
            rng = np.random.default_rng()
            rnd_sel = rng.integers(0, len(df_x), size=max_points)
            x = df_x[phenotype].iloc[rnd_sel].values
            y = df_y[phenotype].iloc[rnd_sel].values
            xerr = np.sqrt(df_x[f'{phenotype}_var']).iloc[rnd_sel].values
            if color_by_err != 'experiment':
                yerr = df_y[f'{phenotype}_err'].iloc[rnd_sel].values
                err = np.sqrt(xerr**2 + yerr**2)
            
        if color_by_err == 'combined':
            c = err
        elif color_by_err == 'experiment':
            c = xerr
        elif color_by_err == 'LANTERN':
            c = yerr
            
        if sort_by_err:
            df_plot = pd.DataFrame({'x':x, 'y':y, 'c':c})
            df_plot.sort_values(by='c', ascending=False, inplace=True)
            x = df_plot.x
            y = df_plot.y
            c = df_plot.c
            
        im = ax.scatter(x, y, c=c, cmap=cmap, alpha=alpha)
        
        ylim = ax.get_ylim()
        ax.plot(ylim, ylim, '--k');
        ax.set_xlabel(f'{phenotype}, experiment', size=14)
        ax.set_ylabel(f'{phenotype}, LANTERN', size=14)
        
        if colorbar:
            ax_box = ax.get_position()
            w = ax_box.width/15
            h = ax_box.height
            x = ax_box.x1 + w
            y = ax_box.y0
            cb_ax = fig.add_axes([x, y, w, h])
            cbar = fig.colorbar(im, cax=cb_ax, **cbar_kwargs)
            cbar.solids.set(alpha=1)
            cbar.ax.set_ylabel(f'{color_by_err} error', rotation=270, labelpad=20, size=14)
        
        return fig, ax
    

def invgammalogpdf(x, alpha, beta):
    return alpha * beta.log() - torch.lgamma(alpha) + (-alpha - 1) * x.log() - beta / x

def weighted_rms_residual(x, y, yerr=None, xerr=None):
    x = np.array(x)
    y = np.array(y)
    
    if (yerr is None) and (xerr is None):
        w = 1
    elif yerr is None:
        xerr = np.array(xerr)
        w = 1/xerr**2
    elif xerr is None:
        yerr = np.array(yerr)
        w = 1/yerr**2
    else:
        xerr = np.array(xerr)
        yerr = np.array(yerr)
        w = 1/(xerr**2 + yerr**2)
    
    resid = y - x
    rms = np.sqrt(np.average(resid**2, weights=w))
    
    return rms
