# %%
import pandas as pd
import numpy as np

corrs = np.random.randn(14, 10)
params = np.random.randn(14, 10)

df_corrs = pd.DataFrame(corrs, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
df_params = pd.DataFrame(params, columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# %%
df_corrs
# %%
df_params
# %%
mask = df_corrs.eq(df_corrs.max(axis=1), axis=0)
print(mask)
masked = df_params[mask].values.flatten()
params = masked[masked == masked.astype(float)]
params

# %%
df_corrs.max(axis=1).values
# %%
