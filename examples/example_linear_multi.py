import numpy as np
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c

COMP_MATLAB = True  # Set to True if you want to compare with MATLAB data
DATA_FROM_MATLAB = False  # Set to True if you want to load data from MATLAB files

if __name__ == '__main__':

    wvl0 = 1550e-9
    L0 = 1.0
    n2 = 2.3e-20
    
    beta2 = -0.0123

    num_modes = 1

    Nt = 2**10
    time_window = 10 # ps
    dt = time_window / Nt
    tfwhm = 1.0 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 0.001
    z = np.arange(0, L0, dz)
    Nz = len(z)

    total_energy = 5.0 # nJ
    
    num_modes = 1

    overlap = 1.9454e10
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)
    # omega
    gamma = 2 * n2 * np.pi / wvl0 * overlap


    if DATA_FROM_MATLAB:
        from scipy.io import loadmat
        betas = loadmat('./data/betas.mat')['betas']
        S = loadmat('./data/GRIN_1030_S.mat')['SR']
    else:
        betas = [0, 0, 18.9382, 0, 0, 0]
        S = 0
    

    

    domain = Domain(Nt, Nz, dz, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, gamma=gamma, n2=n2, betas=betas, L=L0, fR=0.)
    initial_fields = Pulse(domain, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian', values=None)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, num_save=100)

    # Input comparison with MATLAB
    if COMP_MATLAB:
        from scipy.io import loadmat
        # load input and output from MATLAB files
        initial_fields_gt = loadmat('./data/initial_pulse.mat')['initial_fields']
        output_fields_gt = loadmat('./data/output_pulse_spm.mat')['output_field']



    # Define simulation
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    
    # Plot the initial pulse, fiber structure
    plt.figure()
    initial_fields_np = initial_fields.fields.cpu().numpy()
    for n in range(num_modes):
        plt.plot(domain.t, np.abs(initial_fields_np[n]), 'r-', label='calculated', alpha=0.8,  linewidth=2.0)
    if COMP_MATLAB:
        plt.plot(domain.t, initial_fields_gt, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
    plt.legend(fontsize=15)


    # Run the simulation
    sim.run()


    output_fields = sim.output_fields.cpu().numpy()
    output_intensity = np.abs(output_fields)**2

    
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
