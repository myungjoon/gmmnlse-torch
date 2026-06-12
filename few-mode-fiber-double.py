import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

import os

plt.rcParams['font.size'] = 15

DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False

NUM_SAVE = 100
NUM_CHUNKS = 10

device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")


def get_hrw(ts, t1=12.2e-3, t2=32e-3):
    hr = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
    hrw = np.fft.ifft(hr) * Nt
    return hrw

if __name__ == '__main__':

    L0 = 1.0
    dz = 1e-4
    
    wvl0 = 1550e-9
    n2 = 2.3e-20
    fr = 0.18

    num_modes = 3
    total_energy = 5.0 # nJ
    
    Nt = 2**12
    time_window = 40 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # s
    tfwhm = 0.250 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    freq_abs = f0 + freq
    wavelength = c0 / freq_abs     
    wavelength_nm = np.sort(wavelength * 1e9)

    ts = np.linspace(0, time_window, Nt)
    t1 = 12.2e-3
    t2 = 32e-3

    
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    hrw = get_hrw(ts)
    hrw = torch.tensor(hrw, dtype=torch.complex128, device=device)

    S = np.load('./fibers/Sk_3modes.npy')
    S = torch.tensor(S, dtype=torch.complex128, device=device)
    # betas = np.load('./fibers/beta_3modes.npy')
    # betas = np.array([
    #      beta0   beta1    beta2        beta3
    #     [  0.0,    0.0,    -2.538e-2,    1.508e-4],   # LP01
    #     [  0.0,    0.2,    -2.551e-2,    1.477e-4],   # LP11a
    #     [  0.0,    0.2,    -2.551e-2,    1.477e-4],   # LP11b
    # ])

    betas = np.array([
        #  beta0   beta1    beta2(×1e3)   beta3(×1e6)
        [  0.0,    0.0,     -25.38*1e-3,       150.8*1e-6],   # LP01
        [  0.0  ,    0.2,     -25.51*1e-3,       147.7*1e-6],   # LP11a
        [  0.0,    0.2,     -25.51*1e-3,       147.7*1e-6],   # LP11b
    ])
   
    betas = torch.tensor(betas, dtype=torch.float64, device=device)
    
    Nz = int(L0 / dz)
    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    coeffs = torch.ones(num_modes, dtype=torch.complex128, device=device,)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    input_fields = initial_fields.fields
    # initial_fields = Pulse(domain, coeffs,  tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=fields, device=device)
    # initial_fields.fields = initial_fields.fields * (79.0445 / 137.0312)
    
    boundary = Boundary('periodic')
    config = SimConfig(dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=NUM_SAVE, num_chunks=NUM_CHUNKS)    # Define simulation
        
    # norm = torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
    # coeffs_opt = coeffs / (norm + 1e-12)
    # pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)

    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    sim.run(requires_grad=False)
    output_fields = sim.fields.fields

    fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))

    input_fields_np = input_fields.detach().cpu().numpy()
    output_fields_np = output_fields.detach().cpu().numpy()
    for i in range(num_modes):
        ax2[0].plot(domain.t, np.abs(input_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    
    ax2[0].legend(fontsize=14)
    ax2[0].set_xlim([-time_window / 2, time_window / 2])
    ax2[0].set_xlabel('Time (ps)', fontsize=20)
    ax2[0].set_ylabel('Intensity (a.u.)', fontsize=20)

    input_spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(input_fields_np, axes=0)))**2)
    for i in range(num_modes):
        ax2[1].plot(wavelength_nm, input_spectrum[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    # ax2[1].set_xlim([700, 1400])
    ax2[1].legend(fontsize=14)
    ax2[1].set_xlim([1300, 1900])
    ax2[1].set_xlabel('Wavelength (nm)', fontsize=25)
    ax2[1].set_ylabel('Intensity (a.u.)', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('few-mode-input.png', dpi=300)

    # Plot current output and ground truth for each mode
    fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
    output_fields_np = output_fields.detach().cpu().numpy()
    for i in range(num_modes):
        ax2[0].plot(domain.t, np.abs(output_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    
    ax2[0].legend(fontsize=14)
    ax2[0].set_xlim([-time_window / 2, time_window / 2])
    ax2[0].set_xlabel('Time (ps)', fontsize=20)
    ax2[0].set_ylabel('Intensity (a.u.)', fontsize=20)

    output_spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(output_fields_np, axes=0)))**2)
    for i in range(num_modes):
        ax2[1].plot(wavelength_nm, output_spectrum[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    ax2[1].set_xlim([1300, 1900])
    ax2[1].legend(fontsize=14)
    ax2[1].set_xlabel('Wavelength (nm)', fontsize=25)
    ax2[1].set_ylabel('Intensity (a.u.)', fontsize=20)
    
    plt.tight_layout()
    plt.savefig('few-mode-output.png', dpi=300)
    # plt.show()