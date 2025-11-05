#%%
from ucimlrepo import fetch_ucirepo
import numpy as np

udata = fetch_ucirepo(id=211) #this is the same dataset, but not normalized
x= udata.data.features
y= udata.data.targets.iloc[:,0]
#%%
#%%

x_un=x[['pctBlack', 'pctWhite', 'pctWdiv', 'pctPubAsst',
       'pctPoverty', 'pctAllDivorc', 'pctKids2Par', 'pctPersOwnOccup']] #unnormalized category names are slightly differene
means = np.mean(x_un, axis=0)
stds = np.std(x_un, axis=0)

def normalize_func(new_values):



    #these bounds were determined by the UCI dataset
    lower_bounds = means - 3 * stds
    upper_bounds = means + 3 * stds

    new_values = np.array(new_values)

    new_values_clipped = np.clip(new_values, lower_bounds, upper_bounds)

    # Compute the min and max values for each column after clipping
    min_vals = np.min(x_un, axis=0)
    max_vals = np.max(x_un, axis=0)

    # Perform column-wise min-max normalization on the input values
    norm_values = (new_values_clipped - min_vals) / (max_vals - min_vals + 1e-8)  # small epsilon to avoid div-by-zero

    return norm_values

#%%
