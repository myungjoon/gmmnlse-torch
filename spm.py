import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from gmmnlse import Pulse, Fiber
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse import run

from gmmnlse.mode import ModeSolver

if __name__ == '__main__':

    c = 299792458
    wvl0 = 1030e-9
    L0 = 1.0 # m
    n2 = 2.3e-20 

    Nt = 2**10
    time_window = 10 # ps
    dt = time_window / Nt
    dz = 0.001
    z = np.arange(0, L0, dz)
    Nz = len(z)
    overlap = 1.9454e10
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)
    
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt)
    
    tfwhm = 1 # ps
    beta2 = -0.0123

    gamma = 2 * n2 * np.pi / wvl0 * overlap
    total_energy = 3 # nJ

    # define fiber
    fiber = Fiber(wvl0=wvl0, gamma=gamma, betas=[0, beta2], L=L0, fR=0.)

    # initial_pulse = Pulse.gaussian(tfwhm, total_energy, t, C=-5)
    initial_pulse = Pulse(tfwhm=tfwhm, total_energy=total_energy, t=t, type='gaussian', C=-5)
    

    # Comparison with the MATLAB data
    file_path = './data/initial_pulse_spm.mat'
    initial_pulse_gt = loadmat(file_path)
    initial_pulse_gt = initial_pulse_gt['initial_field'].squeeze()

    file_path2 = './data/output_pulse_spm.mat'
    output_gt = loadmat(file_path2)
    output_gt = output_gt['output_field'].squeeze()

    # plt.figure()
    # plt.plot(t, initial_pulse_gt, label='groundtruth', color='k', linestyle='--', alpha=0.8, linewidth=2)
    # plt.plot(t, initial_pulse, label='calculated', alpha=0.8,  linewidth=3)
    # plt.legend(fontsize=15)

    # Simulation
    output = run(initial_pulse, fiber, t, omega, Nt, Nz)
    output_intensity = np.abs(output)**2
    # final_output = output[:,-1].reshape(-1, 1)
    # output_gt = output_gt.reshape(-1, 1)
    # final_output = output[:,-1]
    num_save = 100
    # indices = np.linspace(0, len(output_intensity), num_save, dtype=int)

    plt.figure()
    print(np.argmax(omega), np.argmin(omega))
    mode_1 = output[0]
    plt.title('Frequency')
    plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(mode_1)))**2, 'r*', label='calculated', alpha=0.8)
    plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(mode_1)))**2, 'r-', label='calculated', alpha=0.8, linewidth=1)
    plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_gt)))**2, 'k--', label='groundtruth', alpha=0.8)

    plt.xlim([-50, 50])
    # plt.plot(omega, np.abs(np.fft.fft(np.fft.ifftshift(output_gt)))**2, 'k--', label='groundtruth', alpha=0.8)
    plt.legend(fontsize=15)

    plt.figure()
    plt.plot(t, np.abs(mode_1)**2, label='calculated', alpha=0.8)
    plt.plot(t, np.abs(output_gt)**2, label='groundtruth', linestyle='--', alpha=0.8)

    # plot_temporal_evolution(output_intensity[:,indices], extent=[-10, 10, 0, 100], xlim=[-2, 2])
    # plot_spectral_evolution(output_intensity[:,indices], extent=[-10, 10, 0, 100])

    plt.show()
