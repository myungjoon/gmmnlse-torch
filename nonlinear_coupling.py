import numpy as np
import torch
import scipy.io as sio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_modes = 10
Nx, Ny = 400, 400
Lx, Ly = 62.5e-6 ,62.5e-6
# dx, dy = Lx / Nx, Ly / Ny
dx, dy = 3.75e-7, 3.75e-7
print(f'dx : {dx}, dy : {dy}')

wvl0 = 1030e-9

# fields = torch.rand((num_modes, Nx, Ny), dtype=torch.complex128)
filepath = f'data/GRIN_1030/radius20boundary0000fieldscalarmode1wavelength1030.mat'
phi = sio.loadmat(filepath)['phi']
phi_shape = phi.shape
mode_fields = np.zeros((num_modes, phi_shape[0], phi_shape[1]), dtype=np.complex128)
mode_fields[0] = phi
for i in range(1, num_modes):
    filepath = f'data/GRIN_1030/radius20boundary0000fieldscalarmode{i+1}wavelength1030.mat'
    mode_fields[i] = sio.loadmat(filepath)['phi']

# filepath = './GRIN_1030/modes_1030.npy'
# fields = np.load(filepath)

fields = torch.tensor(mode_fields, dtype=torch.complex64, device=device)
fields = fields[:num_modes]

norms = torch.sqrt(torch.sum(torch.abs(fields)**2, dim=(1, 2)))
numerator = torch.einsum('phw, lhw, mhw, nhw -> plmn', fields, fields, fields, fields)

norms_p = norms.view(num_modes, 1, 1, 1)
norms_l = norms.view(1, num_modes, 1, 1)
norms_m = norms.view(1, 1, num_modes, 1)
norms_n = norms.view(1, 1, 1, num_modes)
denominator = norms_p * norms_l * norms_m * norms_n

S_k = numerator / denominator
S_k = S_k / (dx * dy)

S_k_numpy = S_k.cpu().numpy()

# thresholding
threshold = np.max(np.abs(S_k_numpy)) / 1e7
S_k_numpy[np.abs(S_k_numpy) < threshold] = 0

# load mat file
S_k_matlab = sio.loadmat('data/GRIN_1030/S_tensors_10modes.mat')
S_k_matlab = S_k_matlab['SR']

# Compare with MATLAB result
norm_matlab = np.linalg.norm(S_k_matlab)
norm_error = np.linalg.norm(S_k_numpy - S_k_matlab)

print(f'Number of modes: {num_modes}')
print(f'Norm of MATLAB result : {norm_matlab}')
print(f'Norm of difference : {norm_error}')

print(f'Ratio of norms : {norm_error / norm_matlab}')

np.save(f'S_k_{num_modes}modes.npy', S_k_numpy)