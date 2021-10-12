#%%
from csng_invariances.models.discriminator import get_core_trained_model
from csng_invariances.datasets.lurz2020 import get_dataloaders

dataloaders = get_dataloaders()
model = get_core_trained_model(dataloaders)
print(model)

# %%
