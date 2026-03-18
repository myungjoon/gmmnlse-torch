import numpy as np
import matplotlib.pyplot as plt
import torch

from gmmnlse import Domain, Pulse, Fiber, Boundary, ModeSolver, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution, plot_intensity, plot_mode_energy_evolution


COMP_MATLAB = False  # Set to True if you want to compare with MATLAB data
DATA_FROM_MATLAB = False  # Set to True if you want to load data from MATLAB files

KERR = False
RAMAN = False
SELF_STEEPING = False
DISORDER = False
GAIN = False
LOSS = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

precision = 'single'
print(f'precision: {precision}')
if precision == 'double':
    real_type = torch.float64
    complex_type = torch.complex128
elif precision == 'single':
    real_type = torch.float32
    complex_type = torch.complex64
else:
    raise ValueError(f'Invalid precision: {precision}')


if __name__ == '__main__':
    
    num_save = 50
    wvl0 = 1030e-9
    L0 = 0.01
    
    num_modes = 6

    # pulse
    total_energy = 10.0 # nJ
    Nt = 2**11
    time_window = 10 # ps
    dt = time_window / Nt
    tfwhm = 1.0 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 1e-5
    z = np.arange(0, L0, dz)
    Nz = len(z)

    # Fiber parameters
    core_radius = 16.0e-6 / 2
    NA = 0.14
    n_core = 1.45
    n_clad = np.sqrt(n_core**2 - NA**2)

    # Simulation domain parameters
    Lx, Ly = 4 * core_radius, 4 * core_radius
    unit = 1e-6
    Nx, Ny = 64, 64
    print(f'The grid size is {Nx}x{Ny}')

    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)

    # beta
    betas = np.load('betas.npy')
    betas[2] = betas[2] * 1e-3
    betas[3] = betas[3] * 1e-6
    betas = betas.T
    
    betas = torch.tensor(betas, dtype=real_type, device=device)
    print(f'betas: {betas}')
    
    # nonlinear coupling
    S = np.load('Sk_6modes.npy')
    S = torch.tensor(S, dtype=complex_type, device=device)
        
    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, betas=betas, L=L0, fr=0., S=S)
    boundary = Boundary('periodic')
    config = SimConfig(num_save=100, raman=RAMAN, kerr=KERR, self_steeping=SELF_STEEPING, disorder=DISORDER, gain=GAIN, loss=LOSS)

    # mode
    # mode_solver = ModeSolver(fiber, num_modes=num_modes, target_mode_indices=list(range(num_modes)), dtype=complex_type)
    # modes_fields, modes_neffs = mode_solver.solve()
    mode_fields = np.load('modes.npy')
    mode_fields = torch.tensor(mode_fields, dtype=complex_type, device=device)

    # input fields
    # coeffs = torch.randn(num_modes, dtype=real_type, device=device)
    coeffs = torch.ones(num_modes, dtype=complex_type, device=device)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian',)

    input_fields = initial_fields.fields.detach().cpu().numpy()

    # Plot the initial pulse, fiber structure# Plot the initial pulse, fiber structure
    
     # Define amd run the simulation
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    sim.run()

    saved_fields = sim.saved_fields.detach().cpu().numpy()
    output_fields = sim.fields.fields.detach().cpu().numpy()
    mode_fields = mode_fields.detach().cpu().numpy()

    output_intensity = np.abs(output_fields)**2
    
    plt.figure(2)
    for n in range(num_modes):
        plt.plot(domain.t, np.abs(input_fields[n]), label=f'mode {n+1}', alpha=0.8, linewidth=2.0)
    plt.legend(fontsize=15)

    plt.figure(3)
    for i in range(num_modes):
        plt.plot(t, output_intensity[i], label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    plt.legend(fontsize=15)
    # plt.figure(2)
    # print(np.argmax(omega), np.argmin(omega))
    # plt.title('Frequency')
    # for i in range(num_modes):
    #     plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_fields[n])))**2, alpha=0.8, linewidth=2.0)

    # plt.xlim([-50, 50])
    # # plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_gt)))**2, 'k--', label='groundtruth', alpha=0.8)
    # plt.legend(fontsize=15)

    plot_mode_energy_evolution(saved_fields, dz=L0/num_save*1e3)

    extent = [-Lx/2, Lx/2, -Ly/2, Ly/2]
    plot_intensity(input_fields, mode_fields, radius=core_radius, extent=extent, title='Input Field')
    plot_intensity(output_fields, mode_fields, radius=core_radius, extent=extent, title='Output Field')


    plt.show()
