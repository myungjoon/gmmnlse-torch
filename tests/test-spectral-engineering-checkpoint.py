import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

import os
import time

plt.rcParams['font.size'] = 15

DATA_PREDEFINED = True  # Set to True if you want to load data from MATLAB files

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

NUM_SAVE = 20
NUM_CHUNKS = 1
USE_CP = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def objective_function_one_wavelength(output_fields, target_mode_index=0, target_wvl_index=0):
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

    intensity = output_spectrum[target_mode_index, target_wvl_index]
    loss = -intensity
    return loss

def objective_over_length(fields, target_mode_index=0, target_wvl_index=0):
    output_spectrum = torch.fft.fftshift(
        torch.abs(torch.fft.fft(torch.fft.ifftshift(fields, dim=-1))) ** 2,
        dim=-1
    )

    output_spectrum = output_spectrum[:, target_mode_index, target_wvl_index]
    opt_length = torch.argmax(output_spectrum)
    
    # Initialize tau with a default value
    tau = torch.tensor(1.0, device=output_spectrum.device, dtype=output_spectrum.dtype)
    
    # Check if we have variation in the spectrum
    max_val = torch.max(output_spectrum)
    min_val = torch.min(output_spectrum)
    delta = max_val - min_val
    
    # Only use adaptive tau if there's sufficient variation
    if delta > 1e-6:  # Use a small threshold instead of checking unique values
        tau = 3.0 / delta

    logsumexp = (1.0 / tau) * torch.logsumexp(tau * output_spectrum, dim=0)
    loss = -logsumexp
    return loss, opt_length

def objective_function_flatness(fields, start=None, end=None):
    #flatness of the spectrum
    spectrum = torch.fft.fftshift(torch.abs(torch.fft.fft(torch.fft.ifftshift(fields, dim=-1))) ** 2, dim=-1)
    flatness = torch.mean(spectrum)
    return flatness

def objective_raman_intensity(fields):
    pass

if __name__ == '__main__':

    L0 = 1.0
    dz = 5e-5
    
    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.18

    num_modes = 8
    total_energy = 300.0 # nJ
    
    Nt = 2**13
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

    target_wvl = 1250
    target_index = np.argmin(np.abs(wavelength_nm - target_wvl))
    print(f"Target index: {target_index}")

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if DATA_PREDEFINED:
        filename = f'./data/predefined_data4.npz'
        S = np.load(filename)['S']
        hrw = np.load(filename)['hrw']
        betas = np.load(filename)['betas']
  
        #convert to torch
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
        S = torch.tensor(S, dtype=torch.complex64, device=device)
        hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)

    Nz = int(L0 / dz)
    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    # coeffs = torch.ones(num_modes, dtype=torch.complex64, device=device, requires_grad=True)
    coeffs = torch.tensor([1.0]+[0.2]*(num_modes-1), dtype=torch.complex64, device=device, requires_grad=True)
    # normalize
    # norm = torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
    # coeffs = coeffs / (norm + 1e-12)
    
    # initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=NUM_SAVE, num_chunks=NUM_CHUNKS)    # Define simulation
        
    lr = 0.01
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    num_iters = 50
    losses = []       

    for it in range(num_iters):
        optimizer.zero_grad(set_to_none=True)

        # Create normalized coefficients without in-place operations
        norm = torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
        coeffs_opt = coeffs / (norm + 1e-12)
        pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)

        sim = Simulation(domain, fiber, pulse, boundary, config)
        start_time = time.time()
        sim.run(use_cp=USE_CP)
        print(f'computation time for forward calculation : {time.time() - start_time}')
        # saved_fields = sim.saved_fields
        output_fields = sim.fields.fields


        loss = objective_function_one_wavelength(output_fields, target_mode_index=0, target_wvl_index=target_index)
        # loss, opt_length = objective_over_length(saved_fields, target_mode_index=0, target_wvl_index=target_index)
        if it == 0:
             # Plot current output and ground truth for each mode
             fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
             # output_fields = saved_fields[opt_length]
             # output_fields = saved_fields[-1]
             output_fields_np = output_fields.detach().cpu().numpy()
             output_intensity = np.abs(output_fields_np)**2
             for i in range(num_modes):
                 line1, = ax2[0].plot(domain.t, output_intensity[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
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
             ax2[1].set_xlabel('Wavelength (nm)', fontsize=25)
             ax2[1].set_ylabel('Intensity (a.u.)', fontsize=20)
             plt.show()
    
        losses.append(loss.item())
        print(f"loss: {loss.item()}")
        print(f"coeffs: {coeffs}")
        # print(f"opt_length_index: {opt_length}")

        start_time = time.time()
        loss.backward()
        print(f'computation time for backpropagation : {time.time() - start_time}')

        # # Compute finite difference gradient for the first coefficient (real part only)
        # fd_eps = 1e-4
        # coeffs_np = coeffs.detach().cpu().numpy()
        # fd_grad = np.zeros_like(coeffs_np, dtype=np.float32)

        # for idx in range(len(coeffs_np)):
        #     # Perturb only the real part of coeffs[idx]
        #     coeffs_perturb = coeffs.clone().detach()
        #     coeffs_perturb[idx] = coeffs_perturb[idx] + fd_eps
        #     # Normalize as in the main loop
        #     norm_perturb = torch.sqrt(torch.sum(torch.abs(coeffs_perturb) ** 2))
        #     coeffs_perturb_opt = coeffs_perturb / (norm_perturb + 1e-12)
        #     pulse_perturb = Pulse(domain, coeffs_perturb_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
        #     sim_perturb = Simulation(domain, fiber, pulse_perturb, boundary, config)
        #     sim_perturb.run()
        #     output_fields_perturb = sim_perturb.fields.fields
        #     loss_perturb = objective_function_one_wavelength(output_fields_perturb, target_mode_index=0, target_wvl_index=target_index)
        #     # Finite difference: only real part
        #     fd_grad[idx] = ((loss_perturb.item() - loss.item()) / fd_eps).real

        # # Get autograd gradient (real part only)
        # if coeffs.grad is not None:
        #     autograd_grad = coeffs.grad.detach().cpu().numpy().real
        # else:
        #     autograd_grad = np.zeros_like(coeffs_np)

        # Print or plot comparison for visualization
        # print("Finite difference gradient (real part):", fd_grad)
        # print("Autograd gradient (real part):", autograd_grad)

        # # Optional: plot for visualization
        # mode_nums = np.arange(len(fd_grad)) + 1
        # plt.figure(figsize=(7,4))
        # plt.plot(mode_nums, fd_grad, 'o-', label='Finite Difference (real)')
        # plt.plot(mode_nums, autograd_grad, 'x-', label='Autograd (real)')
        # plt.xlabel('Coefficient index', fontsize=16)
        # plt.ylabel('Gradient (real part)', fontsize=16)
        # plt.legend(fontsize=14)
        # plt.title('Gradient Comparison: Finite Difference vs Autograd (real part)')
        # plt.tight_layout()
        # plt.show()
        optimizer.step()
        

    # output_fields = saved_fields[opt_length].detach().cpu().numpy()
    # output_fields = saved_fields[opt_length]
    output_fields = output_fields.detach().cpu().numpy()
    output_intensity = np.abs(output_fields)**2

    # Plot the loss curve over iterations
    plt.figure(figsize=(7, 4))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.grid(True)
    plt.tight_layout()

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
