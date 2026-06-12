import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

import os, time

plt.rcParams['font.size'] = 15

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True
DISORDER = False
GAIN = False
LOSS = False

NUM_SAVE = 200

os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

def get_hrw(ts, t1=12.2e-3, t2=32e-3):
    hr = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
    hrw = np.fft.ifft(hr) * Nt
    return hrw

if __name__ == '__main__':

    L0 = 10.0
    dz = 5e-5
    
    wvl0 = 1550e-9
    n2 = 2.3e-20
    fr = 0.18

    num_modes = 91
    total_energy = 200.0 # nJ
    
    Nt = 2**13
    time_window = 100 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # s
    tfwhm = 0.5 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    freq_abs = f0 + freq
    wavelength = c0 / freq_abs     
    wavelength_nm = np.sort(wavelength * 1e9)

    ts = np.linspace(0, time_window, Nt)
    t1 = 12.2e-3
    t2 = 32e-3
   
    hrw = get_hrw(ts)
    hrw = torch.tensor(hrw, dtype=torch.complex128, device=device)

    S = np.load(f'./Sk_{num_modes}modes.npy')
    S = torch.tensor(S, dtype=torch.complex128, device=device)
    betas = np.load(f'./beta_{num_modes}modes.npy')
    betas = torch.tensor(betas, dtype=torch.float64, device=device)
    

    num_modes = 50
    S = S[:num_modes,:num_modes,:num_modes,:num_modes]
    betas = betas[:num_modes]


    Nz = int(L0 / dz) + 1

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    # coeffs = torch.tensor([1.0]*num_modes, dtype=torch.complex128, device=device, requires_grad=True)
    coeffs = torch.zeros(num_modes, dtype=torch.complex128, device=device)
    random_phases = torch.rand(10, device=device) * 2 * torch.pi
    # coeffs[15:25] = torch.exp(1j * random_phases).to(torch.complex128)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    boundary = Boundary('periodic')
    config = SimConfig(dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, disorder=DISORDER, num_save=NUM_SAVE,)

    input_field = initial_fields.fields.cpu().numpy()
    print(f'input field : {input_field.shape}')


    start_time = time.time()
    print('Running simulation...')
    print(f'{num_modes} modes')
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    sim.run(requires_grad=False)
    end_time = time.time()
    print(f'Simulation time: {end_time - start_time} seconds')

    output_fields = sim.fields.fields
    final_field = output_fields.detach().cpu().numpy()

    input_intensity = np.abs(input_field)**2
    final_intensity = np.abs(final_field)**2 
    plt.figure()
    for i in range(num_modes):
        plt.plot(t, final_intensity[i], linewidth=2)
    plt.xlabel('Time (ps)')
    plt.ylabel('Power (W)')
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('time_response.png', dpi=300)

    plt.figure()
    for i in range(num_modes):
        plt.plot(t, input_intensity[i], linewidth=2)
    plt.xlabel('Time (ps)')
    plt.ylabel('Power (W)')
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('time_response_init.png', dpi=300)

    # Spectrum
    field_freq = np.fft.fftshift(np.fft.ifft(final_field, axis=0), axes=0)
    psd = np.abs(field_freq)**2 * c0 / (wavelength_nm * 1e-9)**2

    plt.figure()
    for i in range(num_modes):
        plt.plot(wavelength_nm, psd[i], linewidth=2)
    plt.xlabel('λ (nm)')
    plt.ylabel('PSD (a.u.)')
    # plt.xlim([1500, 1800])
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('spectral_response.png', dpi=300)

    input_freq = np.fft.fftshift(np.fft.ifft(input_field, axis=0), axes=0)
    psd_input = np.abs(input_freq)**2 * c0 / (wavelength_nm * 1e-9)**2
    plt.figure()
    for i in range(num_modes):
        plt.plot(wavelength_nm, psd_input[i], linewidth=2)
    plt.xlabel('λ (nm)')
    plt.ylabel('PSD (a.u.)')
    # plt.xlim([1500, 1800])
    plt.gca().tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig('spectral_response_input.png', dpi=300)

    # output_saved_fields = sim.saved_fields
    # fundamental_fields = output_saved_fields[:,0,:]
    # fundamental_fields = fundamental_fields.detach().cpu().numpy()
    # fundamental_fields_spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(fundamental_fields, axes=-1), axis=-1))**2, axes=-1)
    
    # # 1) λ를 정렬 (필수)
    # idx = np.argsort(wavelength_nm)
    # wavelength_nm_s = wavelength_nm[idx]
    # fundamental_fields_spectrum_s = fundamental_fields_spectrum[..., idx]

    # # 2) 균일 λ grid 만들기
    # Nlam = wavelength_nm_s.size  # 또는 더 크게 (예: 2*N)
    # wavelength_nm_u = np.linspace(wavelength_nm_s[0], wavelength_nm_s[-1], Nlam)

    # # 3) 보간 (axis=-1 방향으로)
    # fundamental_fields_spectrum_u = np.empty(fundamental_fields_spectrum_s.shape[:-1] + (Nlam,), dtype=fundamental_fields_spectrum_s.dtype)
    # for m in range(fundamental_fields_spectrum_s.shape[0]):
    #     fundamental_fields_spectrum_u[m] = np.interp(wavelength_nm_u, wavelength_nm_s, fundamental_fields_spectrum_s[m])


    # fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))
    # # imshow for fundamental_fields
    # ax1[0].imshow(np.abs(fundamental_fields)**2, extent=[-15, 15, 0, 10],aspect='auto', origin='lower', cmap='turbo')
    # ax1[0].set_xticks(np.linspace(-15, 15, 7, endpoint=True))
    # ax1[0].set_yticks(np.linspace(0, 10, 11, endpoint=True))
    # ax1[0].set_xlim([-1, 1])

    # ax1[0].set_xlabel('Time (ps)', fontsize=20)
    # ax1[0].set_ylabel('Distance (cm)', fontsize=20)

    # # ax1[1].plot(wavelength_nm, fundamental_fields_spectrum[0], '-', label=f'mode 1', alpha=0.8, linewidth=2.0)
    # ax1[1].imshow(np.abs(fundamental_fields_spectrum_u)**2, extent=[wavelength_nm_u[0], wavelength_nm_u[-1], 0, 10],aspect='auto', origin='lower', cmap='turbo')
    # ax1[1].set_xlim([1500, 2000])
    # ax1[1].set_xlabel('Wavelength (nm)', fontsize=25)
    # ax1[1].set_ylabel('Intensity (a.u.)', fontsize=20)


    # # Plot current output and ground truth for each mode
    # fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
    # output_fields_np = output_fields.detach().cpu().numpy()
    # for i in range(num_modes):
    #     line1, = ax2[0].plot(domain.t, np.abs(output_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    #     # ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
    
    # ax2[0].legend(fontsize=14)
    # ax2[0].set_xlim([-1, 1])
    # ax2[0].set_xlabel('Time (ps)', fontsize=20)
    # ax2[0].set_ylabel('Intensity (a.u.)', fontsize=20)

    # output_spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(output_fields_np, axes=0)))**2)
    # for i in range(num_modes):
    #     line1, = ax2[1].plot(wavelength_nm, output_spectrum[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    # ax2[1].set_xlim([1500, 2000])
    # ax2[1].legend(fontsize=14)
    # ax2[1].set_xlabel('Wavelength (nm)', fontsize=25)
    # ax2[1].set_ylabel('Intensity (a.u.)', fontsize=20)
    # plt.show()
