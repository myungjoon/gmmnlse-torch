import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

plt.rcParams['font.size'] = 15

COMP_MATLAB = False  # Set to True ibf you want to compare with MATLAB data
DATA_PREDEFINED = True  # Set to True if you want to load data from MATLAB files
DO_OPTIMIZE = False  # Set to True if you want to optimize the initial pulse

DISPERSION = True
KERR = True
RAMAN = False
SELF_STEEPING = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def objective_function(output_fields, target_ind):
#     """
#     Objective function to maximize the peak intensity of the fundamental modeat 1300 nm.
#     """
#     J = torch.sum(torch.abs(output_fields[0, target_ind])**2) 
#     return -J  # We minimize the negative to maximize the peak intensity

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

def generate_random_target_field(domain, num_modes, seed=None, device=None, dtype=torch.float64):
    """
    Generate a random target field as a random sum of modes.

    Args:
        domain: Domain object with .Nt and .t
        num_modes: int, number of modes
        seed: int or None, for reproducibility
        device: torch.device
        dtype: torch dtype

    Returns:
        target_field: torch.Tensor, shape (Nt,), complex
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    # Random complex coefficients for each mode
    coeffs = torch.randn(num_modes, dtype=torch.float64, device=device)
    # coeffs_imag = torch.randn(num_modes, device=device)
    # coeffs = torch.complex(coeffs_real, coeffs_imag)
    coeffs = coeffs / torch.norm(coeffs)

    print(f'Target Coeff : {coeffs}')

    # Use a Gaussian pulse shape for each mode (arbitrary parameters)
    tfwhm = 0.2  # ps
    total_energy = 1.0
    from gmmnlse.fields import Pulse
    pulse = Pulse(domain, coeffs, tfwhm, total_energy=total_energy, type='gaussian')
    fields = pulse.fields  # (num_modes, Nt)
    # Sum over modes to get the target field shape
    # target_field = torch.sum(fields, dim=0)  # (Nt,)
    return fields


if __name__ == '__main__':

    wvl0 = 1030e-9
    L0 = 0.0005
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

    dz = 5e-6
    z = np.arange(0, L0, dz)
    Nz = len(z)

    # FFT 각주파수 (rad/s)
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # [rad/s]
    wvls = 2 * np.pi * c0 / omega
    wvl0 = 1030.0 # nm

    omega0 = 2 * np.pi * c0 / (wvl0 * 1e-9)  # rad/s
    omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # rad/s
    omega_abs = omega0 + omega_rel

    wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

    coeffs = 0.5 + 0.5 * torch.randn(num_modes, dtype=torch.float64, device=device)
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs)**2))

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    initial_pulse = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0,)
    initial_field = initial_pulse.fields
    target_field = generate_random_target_field(domain, num_modes)

    
    import matplotlib.pyplot as plt

    # Plot each mode of the initial_field in a separate graph
    plt.figure()
    for mode_idx in range(initial_field.shape[0]):
        plt.plot(domain.t, initial_field[mode_idx].cpu().numpy(), label=f"Mode {mode_idx} (real)")
        
    plt.xlabel("Time (ps)")
    plt.ylabel("Field Amplitude")
    plt.legend()
    plt.tight_layout()


    plt.figure()
    for mode_idx in range(target_field.shape[0]):
        plt.plot(domain.t, target_field[mode_idx].cpu().numpy(), label=f"Mode {mode_idx} (real)")
        
    plt.xlabel("Time (ps)")
    plt.ylabel("Field Amplitude")
    plt.legend()
    plt.tight_layout()

    num_iters = 1000
    lr = 1e-2

    # Initialize trainable complex coefficients (no positivity constraint)
    theta = torch.randn(num_modes, dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([theta], lr=lr)

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
        output_fields = pulse.fields  # (num_modes, Nt)

        # Compute loss with respect to the target_field
        loss = objective_function(output_fields, target_field)
        if torch.isnan(loss):
            print(f"Loss is nan at iter {it}")
            break

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

    # After optimization, show the result
    with torch.no_grad():
        coeffs_opt = theta / torch.linalg.norm(theta)
        pulse = Pulse(domain, coeffs_opt, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
        output_fields = pulse.fields
        output_sum = torch.sum(output_fields, dim=0)
        output_sum = output_sum / torch.linalg.norm(output_sum)
        target_field_norm = target_field / torch.linalg.norm(target_field)

    # Create a larger figure with explicit size to accommodate all the data
    plt.figure(figsize=(14, 8))
    # Use the same color for each mode's target and output by specifying color explicitly
    # Plot normalized fields as in the objective function calculation
    # Normalize each mode's field for fair comparison (L2 norm over time for each mode)
    # Normalize the entire field (all modes and time) to unit L2 norm, as in the objective function
    target_field_normed = target_field / torch.norm(target_field.reshape(-1))
    output_fields_normed = output_fields / torch.norm(output_fields.reshape(-1))
    for mode_idx in range(target_field.shape[0]):
        color = plt.cm.tab10(mode_idx % 10)
        plt.plot(domain.t, torch.abs(target_field_normed[mode_idx].cpu())**2, label=f'Target Field Intensity (Mode {mode_idx+1})', color=color, alpha=0.8)
        plt.plot(domain.t, torch.abs(output_fields_normed[mode_idx].cpu())**2, label=f'Optimized Output Intensity (Mode {mode_idx+1})', linestyle='--', color=color, alpha=0.8)
    plt.xlabel('Time (ps)')
    plt.ylabel('Intensity (a.u.)')
    plt.title('Target vs. Optimized Output Field Intensity')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()