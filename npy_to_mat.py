import numpy as np
import scipy.io as sio

# Load the .npy file
data = np.load('Sk_6modes.npy')

# Save as .mat file
sio.savemat('Sk_6modes.mat', {'SK': data, 'SR': data})

print(f"Converted Sk_6modes.npy to Sk_6modes.mat")
print(f"Data shape: {data.shape}")
print(f"Data dtype: {data.dtype}")
