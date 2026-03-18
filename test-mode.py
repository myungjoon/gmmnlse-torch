import numpy as np
import matplotlib.pyplot as plt
import torch

from gmmnlse import ModeSolver

wvl0 = 1550e-9

radius = 12e-6
Lx, Ly = 4*radius, 4*radius
Nx, Ny = 256, 256
dx, dy = Lx / Nx, Ly / Ny

num_modes = 21

fiber_type = 'GRIN'
n_clad = 1.45
NA = 0.14

mode_solver = ModeSolver(wvl0, radius, Lx, Ly, Nx, Ny, fiber_type=fiber_type, n_clad=n_clad, NA=NA, num_modes=num_modes)
mode_fields = mode_solver.solve()


fig, axes = plt.subplots(2, 3, figsize=(15, 12))
for i in range(2):
    for j in range(3):
        axes[i, j].imshow(np.abs(mode_fields[i*3+j])**2, aspect='auto', cmap='turbo')
        axes[i, j].set_title(f'Mode {i*3+j+1}')
plt.show()