# %%
import numpy as np

from numpy.linalg import pinv
from concurrent.futures import ProcessPoolExecutor

# %%
class LinearFilter:
    def __init__(self, images, responses):
        self.images = images
        self.responses = responses

    def _rigde_regularized_computation(self):
        fil = np.matmul(
            pinv(
                np.matmul(self.images.T, self.images)
                + reg_factor * np.identity(self.dim1 * self.dim2)
            ),
            np.matmul(self.images.T, responses),
        )

    def train(self):
        # idea multiprocess single computation per image
        with ProcessPoolExecutor() as executor:
            pass


# %%
images = np.load(
    "/Users/leongorissen/csng_invariances/data/external/antolik2016/Data/region1/training_inputs.npy"
)
responses = np.load(
    "/Users/leongorissen/csng_invariances/data/external/antolik2016/Data/region1/training_set.npy"
)
# %%
a = LinearFilter(images, responses)
