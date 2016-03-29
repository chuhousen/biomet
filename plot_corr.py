import utils
import exp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pandas.tools.plotting import scatter_matrix

sns.set_style("whitegrid")
sns.set_palette("bright")

def plot_scatter_matrix(df=exp.get_exp1_data()):
    scatter_matrix(df, alpha=0.2, figsize=(12, 12), diagonal='kde')
    plt.tight_layout()
    plt.show()

def plot_heatmap(df=exp.get_exp1_data()):
    corrmat = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corrmat, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    plt.figure(figsize=(12,12))
    sns.heatmap(corrmat, vmax=.8, square=True, mask=mask)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_scatter_matrix()
