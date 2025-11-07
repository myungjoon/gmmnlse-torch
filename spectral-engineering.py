import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

import os

plt.rcParams['font.size'] = 15

DATA_PREDEFINED = True  # Set to True if you want to load data from MATLAB files

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

NUM_SAVE = 50

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def objective_function_one_wavelength(output_fields, target_index):
    """
    Objective function to maximize the intensity of the fundamental mode (mode 0)
    at a specific spectral index (target_index) in the spectral domain.

    Args:
        output_fields: torch.Tensor, shape (num_modes, Nt), complex
        target_index: int, index in the spectral domain to maximize

    Returns:
        loss: torch.Tensor, scalar (negative intensity at [0, target_index])
    """
    
    output_spectrum = torch.fft.fftshift(
        torch.abs(torch.fft.fft(torch.fft.ifftshift(output_fields, dim=-1))) ** 2,
        dim=-1
    )

    intensity = output_spectrum[0, target_index]
    loss = -intensity
    return loss

def objective_function_bandwidth(spectrum, theta=None, kappa=0.1, delta_lambda=1.0, eps=1e-12, use_rel_threshold_db=None):
    S = spectrum
    Smax = torch.maximum(S.max(), torch.tensor(eps, dtype=S.dtype, device=S.device))

    if theta is None:
        if use_rel_threshold_db is None:
            use_rel_threshold_db = -10.0
        scale = 10.0 ** (use_rel_threshold_db / 10.0)
        theta = (Smax * scale).detach() 

    q = torch.sigmoid((S - theta) / kappa)
    bandwidth = q.sum() * delta_lambda
    return -bandwidth

def objective_function_flatness(spectrum, band_mask=None, eps=1e-12):
    S = spectrum
    if band_mask is None:
        w = torch.ones_like(S)
    else:
        w = band_mask.to(dtype=S.dtype)
        if torch.all(w <= 0):
            raise ValueError("band_mask has no active elements.")

    w = w / (w.sum() + eps)                  # 확률 가중
    mean_band = (S * w).sum()                # band 평균
    S_tilde = S / (mean_band + eps)
    flatness = ((S_tilde - 1.0) ** 2 * w).sum()
    return flatness
    


def objective_over_length(fields, target_mode_index=0, target_wvl_index=0):
    output_spectrum = torch.fft.fftshift(
        torch.abs(torch.fft.fft(torch.fft.ifftshift(fields, dim=-1))) ** 2,
        dim=-1
    )

    output_spectrum = output_spectrum[:, target_mode_index, target_wvl_index]
    opt_length = torch.argmax(output_spectrum)
    unique_spectrum = torch.unique(output_spectrum)
    if unique_spectrum.numel() > 1:
        max_val = torch.max(unique_spectrum).item()
        second_val = torch.max(unique_spectrum[unique_spectrum < max_val]).item()
        delta = max_val - second_val
        tau = 3.0 / delta


    logsumexp = (1.0 / tau) * torch.logsumexp(tau * output_spectrum, dim=0)
    loss = -logsumexp
    return loss, opt_length




if __name__ == '__main__':

    L0 = 0.20
    dz = 1e-5
    
    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.18

    num_modes = 10
    total_energy = 300.0 # nJ
    
    Nt = 2**14
    time_window = 30 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # s
    tfwhm = 0.2 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    freq_abs = f0 + freq
    wavelength = c0 / freq_abs     
    wavelength_nm = np.sort(wavelength * 1e9)

    target_wvl = 1300
    target_index = np.argmin(np.abs(wavelength_nm - target_wvl))
    print(f"Target index: {target_index}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if DATA_PREDEFINED:
        S = np.load('./data/predefined_data.npz')['S']
        hrw = np.load('./data/predefined_data.npz')['hrw']
        betas = np.load('./data/predefined_data.npz')['betas']

        #convert to torch
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
        S = torch.tensor(S, dtype=torch.complex64, device=device)
        hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)

    Nz = int(L0 / dz)
    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    # coeffs = torch.ones(num_modes, dtype=torch.float32, device=device)
    coeffs = torch.tensor([1.0]+[1e-4]*9, dtype=torch.complex64, device=device, requires_grad=True)
    
    # initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=NUM_SAVE)    # Define simulation
        
    lr = 0.01
    # theta = torch.rand(num_modes, dtype=torch.complex64, device=device, requires_grad=True) # Now, theta is complex
    # theta = torch.tensor(coeffs, dtype=torch.complex64, device=device, requires_grad=False)
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    num_iters = 20
    losses = []
    


    # torch.autograd.set_detect_anomaly(True)
    for it in range(num_iters):
        optimizer.zero_grad()

        # Create normalized coefficients without in-place operations
        norm = torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
        coeffs_opt = coeffs / (norm + 1e-12)
        pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)

        sim = Simulation(domain, fiber, pulse, boundary, config)
        sim()
        output_fields = sim.output_fields
        fields = sim.saved_fields

        if it == 0:
             # Plot current output and ground truth for each mode
             fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
             output_fields_np = output_fields.detach().cpu().numpy()
             for i in range(num_modes):
                 line1, = ax2[0].plot(domain.t, np.abs(output_fields_np[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
                 # ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
             
             ax2[0].legend(fontsize=14)
             ax2[0].set_xlim([-1, 1])
             ax2[0].set_xlabel('Time (ps)', fontsize=20)
             ax2[0].set_ylabel('Intensity (a.u.)', fontsize=20)
     
             output_spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(output_fields_np, axes=0)))**2)
             for i in range(num_modes):
                 line1, = ax2[1].plot(wavelength_nm, output_spectrum[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
             ax2[1].set_xlim([700, 1400])
             ax2[1].legend(fontsize=14)
             ax2[1].axvline(x=target_wvl, color='red', linestyle='--', linewidth=2.0)
             ax2[1].set_xlabel('Wavelength (nm)', fontsize=20)
             ax2[1].set_ylabel('Intensity (a.u.)', fontsize=20)
             plt.show()

        loss, opt_length = objective_over_length(fields, target_mode_index=0, target_wvl_index=target_index)
        losses.append(loss.item())
        print(f"loss: {loss.item()}")
        print(f"coeffs: {coeffs}")
        print(f"opt_length_index: {opt_length}")
        
        loss.backward()
        optimizer.step()
        

    # After optimization, show the result
    # with torch.no_grad():
    #     coeffs_opt = theta / torch.linalg.norm(theta)
    #     pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
    #     output_fields = pulse.fields

    # output_fields = sim.output_fields.detach().cpu().numpy()
    output_fields = sim.saved_fields[opt_length].detach().cpu().numpy()
    output_intensity = np.abs(output_fields)**2

    # Plot the loss curve over iterations
    plt.figure(figsize=(7, 4))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    # Plot output pulse vs groundtruth as a separate figure
    # Plot both time and spectral domain comparisons between optimized and ground truth results
    fig_out, ax_out = plt.subplots(1, 2, figsize=(14, 6))

    # --- Time domain subplot ---
    for i in range(num_modes):
        line1, = ax_out[0].plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    ax_out[0].set_xlim([-1, 1])
    ax_out[0].set_xlabel('Time (ps)', fontsize=20)
    ax_out[0].set_ylabel('Intensity (a.u.)', fontsize=20)
    ax_out[0].legend(fontsize=15)
    ax_out[0].set_title('Time Domain', fontsize=18)

    spectrum_opt = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(output_fields, axes=0)))**2)

    for i in range(num_modes):
        line2, = ax_out[1].plot(wavelength_nm, spectrum_opt[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    
    ax_out[1].set_xlim([700, 1400])
    ax_out[1].axvline(x=target_wvl, color='red', linestyle='--', linewidth=2.0)
    ax_out[1].set_xlabel('Wavelength (nm)', fontsize=20)
    ax_out[1].set_ylabel('Intensity (a.u.)', fontsize=20)
    ax_out[1].legend(fontsize=15)
    ax_out[1].set_title('Spectral Domain', fontsize=18)

    plt.tight_layout()
    plt.show()
