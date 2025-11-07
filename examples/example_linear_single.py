import numpy as np
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c

COMP_MATLAB = False  # Set to True if you want to compare with MATLAB data

if __name__ == '__main__':

    wvl0 = 1550e-9
    L0 = 1.0
    n2 = 2.3e-20
    
    beta2 = -0.0123

    num_modes = 1

    Nt = 2**10
    time_window = 80 # ps
    dt = time_window / Nt
    tfwhm = 0.1 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 0.001
    z = np.arange(0, L0, dz)
    Nz = len(z)

    total_energy = 1.0 # nJ
    
    overlap = 1.9454e10
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)
    gamma = 2 * n2 * np.pi / wvl0 * overlap
    
    domain = Domain(Nt=Nt, Nz=Nz, dz=dz, time_window=time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, gamma=gamma, betas=[0, beta2], L=L0, fR=0.)
    initial_fields = Pulse(tfwhm=0, total_energy=0, t=0, p=1, C=0, t_center=0, type='custom', values=None)
    boundary = Boundary('periodic')

    # Input comparison with MATLAB
    if COMP_MATLAB:
        from scipy.io import loadmat
        # load input and output from MATLAB files
        # initial_fields_gt = loadmat('input_pulse.mat')['pulse']
        # output_fields_gt



    # Define simulation
    sim = Simulation(domain, fiber, initial_fields, boundary,)
    
    # Plot the initial pulse, fiber structure
    plt.figure()
    plt.plot(t, initial_fields, 'r-', label='calculated', alpha=0.8,  linewidth=2.0)
    if COMP_MATLAB:
        plt.plot(t, initial_fields_gt, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
    plt.legend(fontsize=15)


    # Run the simulation
    sim.run()


    output_fields = sim.output_fields.cpu().numpy()
    output_intensity = np.abs(output_fields)**2

    # initial_pulse = Pulse.gaussian(tfwhm, total_energy, t, C=-5)
    initial_pulse = Pulse(tfwhm=tfwhm, total_energy=total_energy, t=t, type='gaussian',)
    num_save = 100
    
    plt.figure()
    print(np.argmax(omega), np.argmin(omega))
    plt.title('Frequency')
    plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_fields)))**2, 'r*', label='calculated', alpha=0.8)
    plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_fields)))**2, 'r-', label='calculated', alpha=0.8, linewidth=2.0)
    if COMP_MATLAB:
        plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_fields_gt)))**2, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)

    plt.xlim([-50, 50])
    # plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_gt)))**2, 'k--', label='groundtruth', alpha=0.8)
    plt.legend(fontsize=15)

    plt.figure()
    plt.plot(t, output_intensity, label='calculated', alpha=0.8)
    if COMP_MATLAB:
        plt.plot(t, np.abs(output_fields_gt)**2, label='groundtruth', linestyle='--', alpha=0.8)

    plt.show()
