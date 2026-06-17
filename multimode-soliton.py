import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

import os, time

plt.rcParams['font.size'] = 15

def get_hrw(ts, t1=12.2e-3, t2=32e-3, tb=96e-3,
              fa=0.75, fb=0.21, fc=0.04):
      ha_model = ((t1**2 + t2**2)/(t1*t2**2)) * np.sin(ts/t1) * np.exp(-ts/t2)
      hb_model = ((2*tb - ts)/tb**2) * np.exp(-ts/tb)
      ha = fa*ha_model + (fc*ha_model + fb*hb_model)   # scalar + linear pol
      hrw = np.fft.ifft(ha) * Nt
      return hrw

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

NUM_SAVE = 100

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':

    start_time = time.time()

    np.random.seed(42)
    num_modes = 40
    print(f'number of modes : {num_modes}')

    L0 = 10.0
    dz = 5e-5
    
    wvl0 = 1550e-9
    n2 = 2.3e-20
    fr = 0.245

    
    total_energy = 200.0 # nJ
    
    Nt = 2**13
    time_window = 100 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # s
    tfwhm = 0.250 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)
    t_center = -30

    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    f = f0 + np.arange(-Nt//2, Nt//2) / (Nt * dt_s)   
    wl = c0 / f * 1e9                               
    order = np.argsort(wl)      
    wl_sorted = wl[order]

    ts = dt * np.arange(Nt)
    hrw = get_hrw(ts)
    hrw = torch.tensor(hrw, dtype=torch.complex128, device=device)

    S = np.load('./fibers/S_40modes.npy')
    S = S[:num_modes,:num_modes,:num_modes,:num_modes]
    S = torch.tensor(S, dtype=torch.complex128, device=device)

    betas = np.load('./fibers/betas_40modes.npy')
    betas = betas[:num_modes]
    betas = torch.tensor(betas, dtype=torch.float64, device=device)

    Nz = int(L0 / dz)

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    # amplitudes = torch.ones(num_modes, dtype=torch.float64, device=device,)
    # phases = torch.rand(num_modes, dtype=torch.float64, device=device,) * 2 * torch.pi
    # coeffs = amplitudes * torch.exp(1j * phases)
    # coeffs[:5] = 0
    # coeffs[15:] = 0


    # np.save(f'coeffs_{num_modes}_{dz}.npy', coeffs.cpu().numpy())
    
    coeffs = np.load(f'coeffs_{num_modes}_{dz}.npy')
    coeffs = torch.tensor(coeffs, dtype=torch.complex128, device=device)
    
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=t_center, type='gaussian')
    
    input_fields = initial_fields.fields
    input_fields_np = input_fields.detach().cpu().numpy()
    boundary = Boundary('periodic')
    config = SimConfig(dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=NUM_SAVE)    # Define simulation
  
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    sim.run()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total simulation time : {elapsed_time}')
    output_fields = sim.fields.fields
    total_fields = sim.saved_fields
    total_fields = total_fields.detach().cpu().numpy()
    
    output_fields_np = output_fields.detach().cpu().numpy()

    np.save(f'input_fields_{num_modes}_{dz}.npy', input_fields_np)
    np.save(f'output_fields_{num_modes}_{dz}.npy', output_fields_np)
    
    z = np.linspace(0, L0, NUM_SAVE+1)
    intensity_t = np.abs(total_fields) ** 2
    
    # spectrum
    spec = np.fft.fftshift(
      np.abs(np.fft.fft(np.fft.ifftshift(total_fields, axes=-1), axis=-1))**2,
      axes=-1,
    )
    
    fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))
    
    for i in range(num_modes):
        line1, = ax1[0].plot(domain.t, np.abs(input_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    
    # ax1[0].legend(loc='upper left',fontsize=12)
    # ax1[0].set_xlim([-15, 15])
    ax1[0].set_xlabel('Time (ps)', fontsize=18)
    ax1[0].set_ylabel('Intensity (a.u.)', fontsize=18)

    input_spectrum = np.fft.fftshift(
      np.abs(np.fft.ifft(np.fft.ifftshift(input_fields_np, axes=-1), axis=-1))**2,
      axes=-1,
    )   
    for i in range(num_modes):
        ax1[1].plot(wl_sorted, input_spectrum[i][..., order], '-', label=f'mode {i+1}',)
    # ax2[1].set_xlim([700, 1400])
    # ax1[1].legend(loc='upper left', fontsize=12)
    ax1[1].set_xlabel('Wavelength (nm)', fontsize=18)
    ax1[1].set_ylabel('Intensity (a.u.)', fontsize=18)
    plt.savefig(f'input-{num_modes}-{dz}.png', dpi=300)

    # Plot current output and ground truth for each mode
    fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
    for i in range(num_modes):
        line1, = ax2[0].plot(domain.t, np.abs(output_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
        # ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
    
    # ax2[0].legend(loc='upper left', fontsize=12)
    # ax2[0].set_xlim([-15, 15])
    ax2[0].set_xlabel('Time (ps)', fontsize=18)
    ax2[0].set_ylabel('Intensity (a.u.)', fontsize=18)

    output_spectrum = np.fft.fftshift(
      np.abs(np.fft.ifft(np.fft.ifftshift(output_fields_np, axes=-1), axis=-1))**2,
      axes=-1,
    )   
    for i in range(num_modes):
        line1, = ax2[1].plot(wl_sorted, output_spectrum[i][...,order], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    # ax2[1].legend(loc='upper left', fontsize=12)
    ax2[1].set_xlabel('Wavelength (nm)', fontsize=18)
    ax2[1].set_ylabel('Intensity (a.u.)', fontsize=18)
    plt.savefig(f'output-{num_modes}-{dz}.png', dpi=300)
    
    fig, axes = plt.subplots(num_modes, 2, figsize=(14, 5 * num_modes))
    if num_modes == 1:
        axes = axes[np.newaxis, :]
    
    for m in range(num_modes):
        ax_t = axes[m, 0]
        im_t = ax_t.pcolormesh(
            t, z, intensity_t[:, m, :],
            shading='auto', cmap='turbo',
            vmax=intensity_t.max() * 0.5,
        )
        ax_t.set_xlabel('Time (ps)', fontsize=18)
        ax_t.set_ylabel('Distance z (m)', fontsize=18)
        ax_t.set_title(f'Mode {m} — Temporal', fontsize=18)
        fig.colorbar(im_t, ax=ax_t, label='Intensity (a.u.)')
    
        ax_s = axes[m, 1]
        spec_m = spec[:, m, :][:, order]
        im_s = ax_s.pcolormesh(wl_sorted, z, spec_m, shading='auto', cmap='turbo',)
        ax_s.set_xlabel('Wavelength (nm)', fontsize=18)
        ax_s.set_ylabel('Distance z (m)', fontsize=18)
        ax_s.set_title(f'Mode {m} — Spectral', fontsize=18)
        fig.colorbar(im_s, ax=ax_s, label='Intensity (a.u.)')
    
    plt.tight_layout()
    plt.savefig(f'propagation_map-{num_modes}-{dz}.png', dpi=300)