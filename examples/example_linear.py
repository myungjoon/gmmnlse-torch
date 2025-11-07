import numpy as np
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c

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

    # Input comparison with MATLAB
    if COMP_MATLAB:
        from scipy.io import loadmat



    # Simulation
    sim = Simulation(domain, fiber, input, boundary,)
    sim.run()


    # Multimode setting
    coeff = np.ones((num_modes, 1))
    coeff = coeff / np.sqrt(np.sum(coeff**2, axis=0))  # Normalize coefficients
    print(f'coeff: {coeff}')


    
    

    


    # initial_pulse = Pulse.gaussian(tfwhm, total_energy, t, C=-5)
    initial_pulse = Pulse(tfwhm=tfwhm, total_energy=total_energy, t=t, type='gaussian',)
    num_save = 100
    # Comparison with the MATLAB data

    # plt.figure()
    # plt.plot(t, initial_pulse_gt, label='groundtruth', color='k', linestyle='--', alpha=0.8, linewidth=2)
    # plt.plot(t, initial_pulse, label='calculated', alpha=0.8,  linewidth=3)
    # plt.legend(fontsize=15)

    
    
    
    
    
    output_intensity = np.abs(output)**2
    # final_output = output[:,-1].reshape(-1, 1)
    # output_gt = output_gt.reshape(-1, 1)
    # final_output = output[:,-1]
    
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

    plt.show()
