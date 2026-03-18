import numpy as np
import scipy.io as sio

num_modes = 6
filepath = './GRIN_wavelength1030nm_2/'
wavelength = 10300

# Load first mode to get shape
first_file = filepath + f'mode1wavelength{wavelength}.mat'
first_data = sio.loadmat(first_file)
first_mode = first_data['phi']
mode_shape = first_mode.shape

# Initialize array with correct shape
fields = np.zeros((num_modes, mode_shape[0], mode_shape[1]))

# Load all modes
for i in range(num_modes):
    name = f'mode{i+1}wavelength{wavelength}.mat'
    filename = filepath + name
    data = sio.loadmat(filename)
    output = data['phi']
    fields[i, :, :] = output
    print(f'Loaded {name} with shape {output.shape}')

# Save as modes.npy
np.save('modes.npy', fields)
print(f'\nSaved modes.npy with shape: {fields.shape}')