import numpy as np
import torch
import scipy.io as sio

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.complex128

num_modes = 3
Nx, Ny = 64, 64
Lx, Ly = 32e-6 ,32e-6
dx, dy = Lx / Nx, Ly / Ny
print(f'dx : {dx}, dy : {dy}')

wvl0 = 1030e-9

fields = np.load('modes_FMF.npy')
print(f'fields shape: {fields.shape}')
fields = torch.tensor(fields, dtype=dtype, device=device)

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

# # thresholding
# threshold = np.max(np.abs(S_k_numpy)) / 1e7
# S_k_numpy[np.abs(S_k_numpy) < threshold] = 0

# load mat file
S_k_matlab = sio.loadmat('GRIN_1550_FMF/S_tensors_3modes.mat')
S_k_matlab = S_k_matlab['SR']

# Compare with MATLAB result
norm_matlab = np.linalg.norm(S_k_matlab)
norm_error = np.linalg.norm(S_k_numpy - S_k_matlab)

print(f'Number of modes: {num_modes}')
print(f'Norm of MATLAB result : {norm_matlab}')
print(f'Norm of numpy result : {np.linalg.norm(S_k_numpy)}')
print(f'Norm of difference : {norm_error}')

print(f'Ratio of norms : {norm_error / norm_matlab}')

np.save(f'Sk_{num_modes}modes.npy', S_k_numpy)