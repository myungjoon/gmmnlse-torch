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

if __name__ == '__main__':

    wvl0 = 1030e-9
    L0 = 0.1  # You can increase this to test larger lengths
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


    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
    boundary = Boundary('periodic')
    
    # Use checkpointed simulation for larger L0 values
    use_checkpointing = L0 > 0.05  # Use checkpointing for L0 > 5cm
    checkpoint_segments = max(10, Nz // 100)  # Adaptive number of segments
    
    config = SimConfig(
        center_wavelength=wvl0, 
        dispersion=DISPERSION, 
        kerr=KERR, 
        raman=RAMAN, 
        self_steeping=SELF_STEEPING,
        use_checkpointing=use_checkpointing,
        checkpoint_segments=checkpoint_segments
    )
    
    print(f"Using checkpointing: {use_checkpointing}")
    print(f"Number of segments: {checkpoint_segments}")
    print(f"Total simulation steps: {Nz}")
    
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
    ax_out.set_title(f'Simulation Result (L0={L0}m, Checkpointed={use_checkpointing})', fontsize=16)

    plt.tight_layout()
    plt.show()

    if DO_OPTIMIZE:
        print("Starting optimization...")
        target_field = torch.tensor(output_fields_gt, dtype=torch.complex64, device=device)
        
        # Initialize trainable complex coef
        lr = 0.01
        theta = torch.rand(num_modes, dtype=torch.complex64, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([theta], lr=lr)
        num_iters = 250
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

            # Simple L2 loss for optimization
            loss = torch.mean(torch.abs(output_fields - target_field) ** 2)
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

        # Plot the loss curve over iterations
        plt.figure(figsize=(7, 4))
        plt.plot(losses, label='Loss')
        plt.xlabel('Iteration', fontsize=20)
        plt.ylabel('Loss', fontsize=20)
        plt.title(f'Optimization Loss (L0={L0}m)', fontsize=16)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        print(f"Optimization completed for L0={L0}m!")
        print(f"Final loss: {losses[-1]:.6e}")
