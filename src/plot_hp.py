import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.cluster import KMeans
import ot
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

class plot_hp:
    def __init__(self, trials,int_params_names_to_correlate):

        self.trials = trials  # For storing hyperopt trials
        self.int_params_names_to_correlate=int_params_names_to_correlate

    def trials_plot_param1(self, parname, title, logx=False, linestyle='bs'):
        """Plot trial results against a specified parameter."""
        for trial in self.trials:
            loss_eval = trial['result']['loss']
            param = trial['result']['params'][parname]
            if logx:
                plt.semilogx(np.abs(param), loss_eval, linestyle)
            else:
                plt.plot(param, loss_eval, linestyle)
        plt.xlabel(parname)
        plt.ylabel('loss')
        plt.title(title)
        plt.show()

    def trials_loss_history(self, title):
        """Plot the loss history over trials."""
        plt.figure(figsize=(16, 12))
        loss_eval = [trial['result']['loss'] for trial in self.trials]
        loss_train = [trial['result'].get('loss_train', None) for trial in self.trials]

        plt.plot(loss_train, label="Loss Training Data")
        plt.plot(loss_eval, label="Loss Validation Data")
        best_score = self.trials.best_trial['result']['loss']
        plt.axhline(y=best_score, color='r', linestyle='-')

        plt.xlabel("#Iteration Hyperopt")
        plt.ylabel('Loss')
        plt.title(title)
        plt.legend()
        plt.show()

    def trials_params_history(self, param, title, linestyle='-', logy=False):
        """
        Plot the history of a specific parameter over trials.

        Parameters:
        param: str, the name of the parameter to plot
        title: str, title for the plot
        linestyle: str, line style for the plot
        logy: bool, whether to use a logarithmic scale for the y-axis
        """
        values = [trial['result']['params'][param] for trial in self.trials]

        plt.figure(figsize=(10, 6))
        if logy:
            plt.semilogy(range(len(values)), values, linestyle)
        else:
            plt.plot(range(len(values)), values, linestyle)

        plt.xlabel("#Iteration Hyperopt")
        plt.ylabel(param)
        plt.title(title)
        plt.grid(True)
        plt.show()

    def scatterplot_matrix_colored(self, param_names, param_values, best_accs, title, blur=False):
        """
        Scatterplot colored according to the Z values of the points.

        Parameters:
        param_names: List of parameter names for the axes.
        param_values: List of parameter values.
        best_accs: List of best accuracies to color the points.
        title: Title for the scatterplot.
        blur: Whether to apply blurring effect to points.
        """
        nb_params = len(param_values)
        best_accs = -np.array(best_accs)  # Assuming higher is better
        norm = mcolors.Normalize(vmin=best_accs.min(), vmax=best_accs.max())

        fig, ax = plt.subplots(nb_params, nb_params, figsize=(16, 16))

        for i in range(nb_params):
            p1 = param_values[i]
            for j in range(nb_params):
                p2 = param_values[j]

                axes = ax[i, j]
                s = axes.scatter(p2, p1, s=100, alpha=.3, c=best_accs, cmap='Spectral', norm=norm)
                s = axes.scatter(p2, p1, s=15, c=best_accs, cmap='Spectral', norm=norm)

                if j == 0:
                    axes.set_ylabel(param_names[i], rotation=0)
                else:
                    axes.set_yticks([])

                if i == nb_params - 1:
                    axes.set_xlabel(param_names[j], rotation=90)
                else:
                    axes.set_xticks([])

        fig.subplots_adjust(right=0.82, top=0.95)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(s, cax=cbar_ax)

        plt.suptitle(title)
        plt.show()

    def plot_trials(self):
        """Plot trials to analyze parameters against best accuracy."""

        param_values = [
            [trial['result']['params'][p] for trial in self.trials]
            for p in self.int_params_names_to_correlate
        ]
        best_accs = [trial['result']['loss'] for trial in self.trials]

        self.scatterplot_matrix_colored(self.int_params_names_to_correlate, param_values, best_accs,
                                          title='Scatterplot Matrix of Tried Values in the Search Space Over Different Params',
                                          blur=True)
