import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from gmmnlse import svmodes

field = 'scalar'
boundary = '0000'
wavelength = 775e-9



num_modes = 24
radius = 12e-6
Lx, Ly = 4*radius, 4*radius
Nx, Ny = 2048, 2048
dx, dy = Lx / Nx, Ly / Ny

eps = sio.loadmat('eps.mat')['epsilon']

guess = np.sqrt(eps[Nx//2, Ny//2])

phi, neff = svmodes(wavelength, guess, num_modes, dx, dy, eps, boundary, field)

print('Mode calculation completed')
# transpose the last dimension to the first dimension
phi = np.transpose(phi, (2, 0, 1))


# 4 * 6 mode plots
for i in range(4):
    for j in range(6):
        plt.subplot(4, 6, i*6+j+1)
        plt.imshow(np.abs(phi[i*6+j])**2, aspect='auto', origin='lower', cmap='turbo')
        plt.title(f'Mode {i*6+j+1}')


plt.show()
