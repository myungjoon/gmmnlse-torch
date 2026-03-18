import numpy as np
import scipy.io as sio

# Load betas.mat
data = sio.loadmat('betas.mat')
betas = data['betas']

print(f'Loaded betas.mat with shape: {betas.shape}')

# Save as betas.npy
np.save('betas.npy', betas)
print(f'Saved betas.npy with shape: {betas.shape}')
