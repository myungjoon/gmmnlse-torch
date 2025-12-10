import numpy as np
import torch
import scipy.io as sio
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_modes = 3
Nx, Ny = 64, 64
Lx, Ly = 32e-6 ,32e-6
# dx, dy = Lx / Nx, Ly / Ny
# dx, dy = 3.75e-7, 3.75e-7
dx, dy = Lx / Nx, Ly / Ny
print(f'dx : {dx}, dy : {dy}')
radius = 8e-6

wvl0 = 1550e-9


# filepath = f'./GRIN_1550_FMF/radius{int(radius*1e6)}boundary0000fieldscalarmode1wavelength{int(wvl0*1e9)}.mat'
# phi = sio.loadmat(filepath)['phi']
# phi_shape = phi.shape
# mode_fields = np.zeros((num_modes, phi_shape[0], phi_shape[1]), dtype=np.complex128)
# mode_fields[0] = phi
mode_fields = np.zeros((num_modes, Nx, Ny), dtype=np.complex128)
for i in range(num_modes):
    filepath = f'./GRIN_1550_FMF/radius{int(radius*1e6)}boundary0000fieldscalarmode{i+1}wavelength{int(wvl0*1e9)}.mat'
    mode_fields[i] = sio.loadmat(filepath)['phi']

fields = torch.tensor(mode_fields, dtype=torch.complex64, device=device)

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


# Reference check for MATLAB result
# load mat file
S_k_matlab = sio.loadmat('./GRIN_1550_FMF/S_tensors_3modes.mat')
S_k_matlab = S_k_matlab['SR']

# Compare with MATLAB result
norm_numpy = np.linalg.norm(S_k_numpy)
norm_matlab = np.linalg.norm(S_k_matlab)
norm_error = np.linalg.norm(S_k_numpy - S_k_matlab)

print(f'Number of modes: {num_modes}')
print(f'Norm of numpy result : {norm_numpy}')
print(f'Norm of MATLAB result : {norm_matlab}')
print(f'Norm of difference : {norm_error}')

print(f'Ratio of norms : {norm_error / norm_matlab}')

np.save(f'S_k_{num_modes}modes.npy', S_k_numpy)