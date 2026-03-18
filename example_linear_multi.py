import numpy as np
import matplotlib.pyplot as plt
import torch

from gmmnlse import Domain, Pulse, Fiber, Boundary, ModeSolver, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution


COMP_MATLAB = False  # Set to True if you want to compare with MATLAB data
DATA_FROM_MATLAB = False  # Set to True if you want to load data from MATLAB files


KERR = False
RAMAN = False
SELF_STEEPING = False
DISORDER = False
GAIN = False
LOSS = False

if __name__ == '__main__':
    real_type = torch.float64
    complex_type = torch.complex128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wvl0 = 1030e-9
    L0 = 0.01
    n2 = 2.3e-20
    
    num_modes = 6

    Nt = 2**11
    time_window = 10 # ps
    dt = time_window / Nt
    tfwhm = 1.0 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 5e-5
    z = np.arange(0, L0, dz)
    Nz = len(z)

    total_energy = 5.0 # nJ
    
    overlap = 1.9454e10
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)
    gamma = 2 * n2 * np.pi / wvl0 * overlap


    # beta
    # betas = np.array([[0, 0, 18.9382, 0, 0, 0], [0, 0, 18.9382, 0, 0, 0], [0, 0, 18.9382, 0, 0, 0]])
    betas = np.load('betas.npy')
    betas = betas.T
    betas = torch.tensor(betas, dtype=real_type, device=device)
    
    # nonlinear coupling
    S = np.load('Sk_6modes.npy')
    S = torch.tensor(S, dtype=complex_type, device=device)
        
    domain = Domain(Nt, Nz, dz, dt,time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, gamma=gamma, n2=n2, betas=betas, L=L0, fr=0., S=S)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, num_save=100, raman=RAMAN, kerr=KERR, self_steeping=SELF_STEEPING, disorder=DISORDER, gain=GAIN, loss=LOSS)

    # mode
    # mode_solver = ModeSolver(fiber, num_modes=num_modes, target_mode_indices=list(range(num_modes)), dtype=complex_type)
    # modes_fields, modes_neffs = mode_solver.solve()
    modes_fields = np.load('modes.npy')
    modes_fields = torch.tensor(modes_fields, dtype=complex_type, device=device)


    # input fields
    coeffs = torch.ones(num_modes, dtype=real_type, device=device)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian',)
    
    # Input comparison with MATLAB
    if COMP_MATLAB:
        from scipy.io import loadmat
        # load input and output from MATLAB files
        initial_fields_gt = loadmat('./data/initial_pulse.mat')['initial_fields']
        output_fields_gt = loadmat('./data/output_pulse_spm.mat')['output_field']


    # Define simulation
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    # Run the simulation
    sim.run()

    output_fields = sim.fields.fields.cpu().numpy()
    output_intensity = np.abs(output_fields)**2


    # Plot the initial pulse, fiber structure
    plt.figure(1)
    initial_fields_np = initial_fields.fields.cpu().numpy()
    for n in range(num_modes):
        plt.plot(domain.t, np.abs(initial_fields_np[n]), 'r-', label='calculated', alpha=0.8,  linewidth=2.0)
    if COMP_MATLAB:
        plt.plot(domain.t, initial_fields_gt, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
    plt.legend(fontsize=15)

    plt.figure(2)
    print(np.argmax(omega), np.argmin(omega))
    plt.title('Frequency')
    # for i in range(num_modes):
    #     plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_fields[n])))**2, 'r-', label='calculated', alpha=0.8, linewidth=2.0)

    if COMP_MATLAB:
        plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_fields_gt)))**2, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)

    plt.xlim([-50, 50])
    # plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_gt)))**2, 'k--', label='groundtruth', alpha=0.8)
    plt.legend(fontsize=15)

    plt.figure(3)
    for i in range(num_modes):
        plt.plot(t, output_intensity[n], 'r-', label='calculated', alpha=0.8, linewidth=2.0)
    if COMP_MATLAB:
        plt.plot(t, np.abs(output_fields_gt)**2, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)

    plt.show()
