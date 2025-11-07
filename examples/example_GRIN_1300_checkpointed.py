import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0
from gmmnlse.simulation_checkpointed import SimulationCheckpointed

plt.rcParams['font.size'] = 15

COMP_MATLAB = False  # Set to True if you want to compare with MATLAB data
DATA_PREDEFINED = True  # Set to True if you want to load data from MATLAB files
DO_OPTIMIZE = True  # Set to True if you want to optimize the initial pulse

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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

if __name__ == '__main__':

    wvl0 = 1030e-9
    L0 = 0.1  # Increased from 0.01 to 0.1 (10cm) to test memory efficiency
    n2 = 2.3e-20

    fr = 0.18

    num_modes = 10
    total_energy = 600.0 # nJ
    
    Nt = 2**14
    time_window = 30 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # 초 단위로 변환
    tfwhm = 0.2 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

    dz = 5e-5
    z = np.arange(0, L0, dz)
    Nz = len(z)

    # FFT 각주파수 (rad/s)
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # [rad/s]
    # calculate corresponding lambda
    wvls = 2 * np.pi * c0 / omega

    # Calculate target index for 1300nm
    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    freq_abs = f0 + freq
    wavelength = c0 / freq_abs     
    wavelength_nm = np.sort(wavelength * 1e9)
    target_index = np.argmin(np.abs(wavelength_nm - 1300))
    print(f"Target index: {target_index}")

    coeffs = torch.ones(num_modes, dtype=torch.float32, device=device)
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))

    import os
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
    
    initial_fields_gt = np.load('./data/initial_pulse_all.npy')
    initial_fields_gt = torch.tensor(initial_fields_gt, dtype=torch.complex64, device=device)
    initial_fields_gt = torch.transpose(initial_fields_gt, 0, 1)
    

    output_fields_gt = np.load('./data/output_pulse_all.npy')
    output_fields_gt = np.transpose(output_fields_gt, (1, 0))

    # Test the checkpointed simulation first
    print("Testing checkpointed simulation...")
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, use_checkpointing=True, checkpoint_segments=20)
    
    sim = SimulationCheckpointed(domain, fiber, initial_fields, boundary, config)
    sim.run()
    
    output_fields = sim.output_fields.detach().cpu().numpy()
    output_intensity = np.abs(output_fields)**2
    
    # Plot output pulse vs groundtruth as a separate figure
    fig_out, ax_out = plt.subplots(figsize=(8, 6))
    for i in range(num_modes):
        line1, = ax_out.plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
        ax_out.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
    ax_out.legend(fontsize=15)
    ax_out.set_xlim([-1, 1])
    ax_out.set_xlabel('Time (ps)', fontsize=20)
    ax_out.set_ylabel('Intensity (a.u.)', fontsize=20)
    ax_out.set_title(f'Checkpointed Simulation (L0={L0}m)', fontsize=18)

    plt.tight_layout()
    plt.show()

    if DO_OPTIMIZE:
        print("Starting optimization with checkpointed simulation...")
        target_field = torch.tensor(output_fields_gt, dtype=torch.complex64, device=device)
        
        # Initialize trainable complex coef
        lr = 0.01
        theta = torch.rand(num_modes, dtype=torch.complex64, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        num_iters = 100  # Reduced iterations for testing
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
            sim = SimulationCheckpointed(domain, fiber, pulse, boundary, config)
            sim.run()
            output_fields = sim.output_fields

            if it == 0:
                # Plot current output and ground truth for each mode
                fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
                output_fields_np = output_fields.detach().cpu().numpy()
                for i in range(num_modes):
                    line1, = ax2[0].plot(domain.t, np.abs(output_fields_np[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
                
                ax2[0].legend(fontsize=14)
                ax2[0].set_xlim([-1, 1])
                ax2[0].set_xlabel('Time (ps)', fontsize=20)
                ax2[0].set_ylabel('Intensity (a.u.)', fontsize=20)
                ax2[0].set_title('Initial Time Domain', fontsize=16)

                output_spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(output_fields_np, axes=0)))**2)
                for i in range(num_modes):
                    line1, = ax2[1].plot(wavelength_nm, output_spectrum[i], '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
                ax2[1].set_xlim([700, 1600])
                ax2[1].legend(fontsize=14)
                ax2[1].axvline(x=1300, color='red', linestyle='--', linewidth=2.0, label='Target (1300nm)')
                ax2[1].set_xlabel('Wavelength (nm)', fontsize=20)
                ax2[1].set_ylabel('Intensity (a.u.)', fontsize=20)
                ax2[1].set_title('Initial Spectral Domain', fontsize=16)
                plt.tight_layout()
                plt.show()

            loss = objective_function_1300(output_fields, target_index)
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
                print(f"[{it+1:03d}] Loss: {loss.item():.6e}")
                print("theta:", theta)

        # Plot the loss curve over iterations
        plt.figure(figsize=(7, 4))
        plt.plot(losses, label='Loss')
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.title(f'Optimization Loss (L0={L0}m, Checkpointed)', fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot final results
        output_fields = sim.output_fields.detach().cpu().numpy()
        output_intensity = np.abs(output_fields)**2

        # Plot both time and spectral domain comparisons between optimized and ground truth results
        fig_out, ax_out = plt.subplots(1, 2, figsize=(14, 6))

        # --- Time domain subplot ---
        for i in range(num_modes):
            line1, = ax_out[0].plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
        ax_out[0].set_xlim([-1, 1])
        ax_out[0].set_xlabel('Time (ps)', fontsize=20)
        ax_out[0].set_ylabel('Intensity (a.u.)', fontsize=20)
        ax_out[0].legend(fontsize=15)
        ax_out[0].set_title('Final Time Domain', fontsize=18)

        spectrum_opt = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(output_fields, axes=0)))**2)

        for i in range(num_modes):
            line2, = ax_out[1].plot(wavelength_nm, spectrum_opt[i], '-', label=f'Opt mode {i+1}', alpha=0.8, linewidth=2.0)
        
        ax_out[1].set_xlim([700, 1600])
        ax_out[1].axvline(x=1300, color='red', linestyle='--', linewidth=2.0, label='Target (1300nm)')
        ax_out[1].set_xlabel('Wavelength (nm)', fontsize=20)
        ax_out[1].set_ylabel('Intensity (a.u.)', fontsize=20)
        ax_out[1].legend(fontsize=15)
        ax_out[1].set_title('Final Spectral Domain', fontsize=18)

        plt.tight_layout()
        plt.show()

        print(f"Optimization completed for L0={L0}m using checkpointed simulation!")
        print(f"Final loss: {losses[-1]:.6e}")
        print(f"Memory usage should be significantly reduced compared to non-checkpointed version.")
