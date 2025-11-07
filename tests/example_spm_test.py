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

DISPERSION = False
KERR = True
RAMAN = False
SELF_STEEPING = False

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def compute_spectrum(fields):
    """
    Compute the spectrum for each mode in fields using torch.
    Args:
        fields: torch.Tensor, shape (num_modes, Nt), complex
    Returns:
        spectrum: torch.Tensor, shape (num_modes, Nt), real
    """
    # spectrum = torch.zeros(fields.shape[0], fields.shape[1], dtype=torch.complex128, device=fields.device)
    spectrum = np.zeros((fields.shape[0], fields.shape[1]), dtype=np.complex128)
    fields = fields.cpu().numpy()
    for i in range(fields.shape[0]):
        # spectrum[i] = torch.fft.fftshift(torch.abs(torch.fft.fft(torch.fft.ifftshift(fields[i])))**2)
        spectrum[i] = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(fields[i])))**2)
    # ifftshift along time axis (axis=1)
    # fields_shifted = torch.fft.ifftshift(fields, dim=1)
    # # FFT along time axis
    # spectrum_c = torch.fft.fft(fields_shifted, dim=1)
    # # Power spectrum (intensity)
    # spectrum = torch.abs(spectrum_c) ** 2
    # # fftshift to center zero frequency
    # spectrum = torch.fft.fftshift(spectrum, dim=1)
    # Return as real tensor (float64 if input is float64, else float32)
    return spectrum

def generate_random_target_spectrum(domain, num_modes, seed=None, device=None, dtype=torch.float64):
    """
    Generate a random target spectrum as the spectrum of a random sum of modes.
    Returns:
        target_spectrum: np.ndarray, shape (num_modes, Nt)
    """
    target_fields = generate_random_target_field(domain, num_modes, seed=seed, device=device, dtype=dtype)
    target_spectrum = compute_spectrum(target_fields)
    return target_spectrum

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
    L0 = 0.01
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

    dz = 1e-5
    z = np.arange(0, L0, dz)
    Nz = len(z)

    # FFT 각주파수 (rad/s)
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # [rad/s]
    wvls = 2 * np.pi * c0 / omega
    # wvl0 = 1030.0 # nm

    omega0 = 2 * np.pi * c0 / wvl0   # rad/s
    omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # rad/s
    omega_abs = omega0 + omega_rel

    wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

    coeffs = 0.5 + 0.5 * torch.randn(num_modes, dtype=torch.float64, device=device)
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs)**2))

    S = np.load('./data/predefined_data.npz')['S']
    hrw = np.load('./data/predefined_data.npz')['hrw']
    betas = np.load('./data/predefined_data.npz')['betas']

    #convert to torch
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.complex64, device=device)
    hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    
    initial_fields_gt = np.load('./data/initial_pulse_spm.npy')
    output_fields_gt = np.load('./data/output_pulse_spm.npy')
    initial_fields_gt = torch.tensor(initial_fields_gt, dtype=torch.complex64, device=device)

    initial_fields_gt = torch.transpose(initial_fields_gt, 0, 1)
    output_fields_gt = np.transpose(output_fields_gt, (1, 0))

    initial_pulse = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    initial_field = initial_pulse.fields

    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING)    # Define simulation
    
    sim = Simulation(domain, fiber, initial_pulse, boundary, config)
    sim.run()
    output_field = sim.output_fields
    
    
    # Compute frequency axis (omega) and wavelength axis (wvls_nm)
    Nt = initial_field.shape[1]
    dt = domain.dt
    dt_s = dt * 1e-12  # convert ps to s if needed
    c0 = 299792458  # m/s

    # Frequency axis (rad/s)
    omega = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)
    # Central wavelength (nm)
    wvl0_nm = 1030.0
    omega0 = 2 * np.pi * c0 / (wvl0_nm * 1e-9)
    omega_abs = omega0 + omega
    wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

    # Compute spectrum for each mode
    initial_spectrum = compute_spectrum(initial_field)
    output_spectrum = compute_spectrum(output_field)

    output_field = output_field.detach().cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1 = axes[0]
    ax1.set_title('Initial Pulse', fontsize=16)
    initial_fields_np = initial_fields_gt.detach().cpu().numpy() 
    for i in range(num_modes):
        ax1.plot(domain.t, np.abs(initial_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8,  linewidth=2.0)
 
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Time (ps)', fontsize=14)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=14)

    ax2 = axes[1]
    ax2.set_title('Output Pulse', fontsize=16)
    for i in range(num_modes):
        line1, = ax2.plot(domain.t, np.abs(output_field[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
        ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', label='groundtruth', alpha=0.8, linewidth=1.5, color=line1.get_color())
    ax2.legend(fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_xlabel('Time (ps)', fontsize=14)

    
    plt.show()