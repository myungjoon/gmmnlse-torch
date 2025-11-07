import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

plt.rcParams['font.size'] = 15

DATA_PREDEFINED = True  # Set to True if you want to load data from MATLAB files

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

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


if __name__ == '__main__':

    wvl0 = 1030e-9
    L0 = 0.01
    
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

    dz = 1e-4
    z = np.arange(0, L0, dz)
    Nz = len(z)

    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # [rad/s]
    wvls = 2 * np.pi * c0 / omega


    lambda0_nm = 1030.0 # nm

    omega0 = 2 * np.pi * c0 / (lambda0_nm * 1e-9)  # rad/s
    omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # rad/s
    omega_abs = omega0 + omega_rel

    wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

    # coeffs = 0.01 * torch.ones(num_modes, dtype=torch.float32, device=device)
    # coeffs[0] = 1.0
    # coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs)**2))

    coeffs = torch.ones(num_modes, dtype=torch.float32, device=device)
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))


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
    
       
    initial_fields_gt = np.load('./data/initial_pulse.npy')
    output_fields_gt = np.load('./data/output_pulse_raman_2.npy')
    

    initial_fields_gt = torch.tensor(initial_fields_gt, dtype=torch.complex64, device=device)
    initial_fields_gt = torch.transpose(initial_fields_gt, 0, 1)
    output_fields_gt = np.transpose(output_fields_gt, (1, 0))

    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING)    # Define simulation
    
    
    target_field = torch.tensor(output_fields_gt, dtype=torch.complex64, device=device)

    lr = 0.01
    theta = torch.rand(num_modes, dtype=torch.float32, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)
    num_iters = 500
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
             fig, ax2 = plt.subplots(figsize=(8, 5))
             for i in range(num_modes):
                 line1, = ax2.plot(domain.t, np.abs(output_fields_gt[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
                 # ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
             ax2.legend(fontsize=14)
             ax2.set_xlim([-1, 1])
             ax2.set_xlabel('Time (ps)', fontsize=20)
             ax2.set_ylabel('Intensity (a.u.)', fontsize=20)
             # ax2.set_title('Current Output vs Ground Truth', fontsize=16)
             plt.show()


        # Compute loss with respect to the target_field
        loss = objective_function(output_fields, target_field)
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

    # After optimization, show the result
    with torch.no_grad():
        coeffs_opt = theta / torch.linalg.norm(theta)
        pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
        output_fields = pulse.fields
        output_sum = torch.sum(output_fields, dim=0)
        output_sum = output_sum / torch.linalg.norm(output_sum)
        target_field_norm = target_field / torch.linalg.norm(target_field)


    output_fields = sim.output_fields.detach().cpu().numpy()
    output_intensity = np.abs(output_fields)**2
    
    num_modes = 10

    # Plot the loss curve over iterations
    plt.figure(figsize=(7, 4))
    plt.plot(losses, label='Loss')
    plt.xlabel('Iteration', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    # Plot output pulse vs groundtruth as a separate figure
    fig_out, ax_out = plt.subplots(figsize=(8, 6))
    for i in range(num_modes):
        line1, = ax_out.plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
        ax_out.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())
    ax_out.legend(fontsize=15)
    ax_out.set_xlim([-1, 1])
    ax_out.set_xlabel('Time (ps)', fontsize=20)
    ax_out.set_ylabel('Intensity (a.u.)', fontsize=20)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axes[0]
    ax1.set_title('Initial Pulse', fontsize=16)
    initial_fields_np = initial_fields.fields.detach().cpu().numpy()
    for i in range(num_modes):
        ax1.plot(domain.t, np.abs(initial_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8,  linewidth=2.0)
 
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Time (ps)', fontsize=14)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=14)

    ax2 = axes[1]
    ax2.set_title('Output Pulse', fontsize=16)
    for i in range(num_modes):
        line1, = ax2.plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
        ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', label='groundtruth', alpha=0.8, linewidth=1.5, color=line1.get_color())
    ax2.legend(fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_xlabel('Time (ps)', fontsize=14)

    plt.tight_layout()
    plt.show()
