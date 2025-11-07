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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def objective_function(output_fields, target_field):
    """
    Objective function to compare the output field with a specific target field shape.
    The target field shape is defined as a random sum of modes.

    Args:
        output_fields: torch.Tensor, shape (num_modes, Nt), complex
        target_field: torch.Tensor, shape (Nt,), complex

    Returns:
        loss: torch.Tensor, scalar
    """
    # Normalize both fields for fair comparison
    # output_sum = torch.sum(output_fields, dim=0)  # (Nt,)
    # Normalize output_fields and target_field so that the total intensity (sum of |field|^2 over all modes and time) is 1
    output_norm = torch.norm(output_fields.reshape(-1))
    output_fields = output_fields / output_norm
    target_norm = torch.norm(target_field.reshape(-1))
    target_field = target_field / target_norm

    # Use L2 loss (MSE) between the output and target field
    loss = torch.mean(torch.abs(output_fields - target_field) ** 2)
    return loss


def objective_function_spectrum(output_fields, target_field):
    """
    Objective function to compare the output field with a specific target field shape in the spectral domain.
    The target field shape is defined as a random sum of modes.

    Args:
        output_fields: torch.Tensor, shape (num_modes, Nt), complex
        target_field: torch.Tensor, shape (Nt,), complex

    Returns:
        loss: torch.Tensor, scalar
    """
    # Transform to spectral domain using FFT
    # Apply fftshift to center the spectrum, then FFT, then fftshift back
    output_spectrum = torch.fft.fftshift(torch.abs(torch.fft.fft(torch.fft.ifftshift(output_fields, dim=-1)))**2, dim=-1)
    target_spectrum = torch.fft.fftshift(torch.abs(torch.fft.fft(torch.fft.ifftshift(target_field, dim=-1)))**2, dim=-1)
    
    # Normalize both spectra for fair comparison
    output_norm = torch.norm(output_spectrum.reshape(-1))
    output_spectrum = output_spectrum / output_norm
    target_norm = torch.norm(target_spectrum.reshape(-1))
    target_spectrum = target_spectrum / target_norm

    # Use L2 loss (MSE) between the output and target spectrum
    loss = torch.mean(torch.abs(output_spectrum - target_spectrum) ** 2)
    return loss

def objective_function_1300(output_fields, target_index):
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

def objective_function_total_energy(output_fields, target_index):
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

    intensity = torch.sum(output_spectrum[:, target_index])
    loss = -intensity
    return loss


if __name__ == '__main__':

    
    L0 = 0.015
    dz = 7.5e-5
    
    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.18

    num_modes = 10
    total_energy = 600.0 # nJ
    
    Nt = 2**14
    time_window = 30 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # s
    tfwhm = 0.2 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    
    z = np.arange(0, L0, dz)
    Nz = len(z)

  
    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    freq_abs = f0 + freq
    wavelength = c0 / freq_abs     
    wavelength_nm = np.sort(wavelength * 1e9)

    target_wvl = 1100
    target_index = np.argmin(np.abs(wavelength_nm - target_wvl))
    print(f"Target index: {target_index}")

    coeffs = torch.ones(num_modes, dtype=torch.float32, device=device)
    # coeffs = torch.tensor([1.0]+[1e-4]*9, dtype=torch.complex64, device=device)
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
    

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if DATA_PREDEFINED:
        S = np.load('./data/predefined_data.npz')['S']
        hrw = np.load('./data/predefined_data.npz')['hrw']
        betas = np.load('./data/predefined_data.npz')['betas']

        #convert to torch
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
        S = torch.tensor(S, dtype=torch.complex64, device=device)
        hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
       
    # initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING)    # Define simulation
        
    lr = 0.01
    # theta = torch.rand(num_modes, dtype=torch.complex64, device=device, requires_grad=True) # Now, theta is complex
    theta = torch.tensor(coeffs, dtype=torch.complex64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    num_iters = 200
    losses = []

    for it in range(num_iters):
        optimizer.zero_grad()

        # Add a small epsilon to avoid division by zero in normalization
        norm = torch.linalg.norm(theta)
        if norm.item() < 1e-12 or torch.isnan(norm):
            print(f"Warning: norm of theta is too small or nan at iter {it}, norm={norm.item()}")
            break
        coeffs_opt = theta / (norm + 1e-12)

        # Generate the initial fields with current coeffs
        pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
        sim = Simulation(domain, fiber, pulse, boundary, config)
        sim.run()
        output_fields = sim.output_fields

        if it == 0:
             # Plot current output and ground truth for each mode
             fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
             output_fields_np = output_fields.detach().cpu().numpy()
             for i in range(num_modes):
                 line1, = ax2[0].plot(domain.t, np.abs(output_fields_np[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
                 # ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
            #  Represent target index
             
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

        loss = objective_function_1300(output_fields, target_index)
        print(f"[{it+1:03d}] Loss: {loss.item():.6e}")
        loss.backward()

        # Check for nan in gradients
        if torch.isnan(theta.grad).any():
            print(f"theta.grad became nan at iter {it}")
            print("theta:", theta)
            print("theta.grad:", theta.grad)
            break

        optimizer.step()

        losses.append(loss.item())
        if (it+1) % 10 == 0 or it == 0:
            
            print("theta:", theta)

    # After optimization, show the result
    # with torch.no_grad():
    #     coeffs_opt = theta / torch.linalg.norm(theta)
    #     pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
    #     output_fields = pulse.fields

    output_fields = sim.output_fields.detach().cpu().numpy()
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
