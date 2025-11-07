import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0
# from gmmnlse.simulation_checkpointed import SimulationCheckpointed
from gmmnlse.simulation import Simulation
import os

plt.rcParams['font.size'] = 15

DATA_PREDEFINED = True  # Set to True if you want to load data from MATLAB files
DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    # You can increase L0 to test larger fiber lengths
    L0 = 0.045  # Increased from 0.01 to 0.1 (10cm) to demonstrate checkpointing
    dz = 1e-5
    
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

    target_index = np.argmin(np.abs(wavelength_nm - 1300))
    print(f"Target index: {target_index}")
    print(f"Fiber length L0: {L0}m")
    print(f"Number of simulation steps: {Nz}")


    # Make coeffs tensor
    coeffs = torch.tensor([0.973164+0.060183j, -0.010206+0.073658j, -0.074634-0.096704j, -0.009493-0.001799j, 0.040928+0.006726j, -0.088571-0.100660j, 0.042284-0.021781j, 0.062423-0.021123j, -0.019264-0.014601j, -0.001594+0.043529j], dtype=torch.complex64, device=device)
    coeffs = torch.abs(coeffs)
    # coeffs = torch.tensor([0.3746+0.0563j, 0.0597+0.0332j, 0.2690+0.4243j, 0.2536+0.3879j,
    #     0.2342+0.1793j, 0.2037+0.2761j, 0.1439+0.3746j, 0.0628+0.0233j,
    #     0.2674+0.2669j, 0.3037+0.1363j], dtype=torch.complex64, device=device)
    # coeffs with mode 0 is 1 and others are 1e-4 and normalize
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
       
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    boundary = Boundary('periodic')
    
    config = SimConfig(
        center_wavelength=wvl0, 
        dispersion=DISPERSION, 
        kerr=KERR, 
        raman=RAMAN, 
        self_steeping=SELF_STEEPING,
    )
    
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    sim.run()
    output_fields = sim.output_fields

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