import numpy as np
import torch
import matplotlib.pyplot as plt

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

plt.rcParams['font.size'] = 15

COMP_MATLAB = False  # Set to True if you want to compare with MATLAB data
DATA_FROM_MATLAB = True  # Set to True if you want to load data from MATLAB files
DO_OPTIMIZE = True  # Set to True if you want to optimize the initial pulse

DISPERSION = True
NONLINEAR = True
RAMAN = True
SELF_STEEPING = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective_function(output_fields, target_ind):
    """
    Objective function to maximize the peak intensity at 1300 nm.
    """
    J = torch.sum(torch.abs(output_fields[:, target_ind])**2)
    return -J  # We minimize the negative to maximize the peak intensity

if __name__ == '__main__':

    wvl0 = 1030e-9
    L0 = 0.001
    n2 = 2.3e-20

    # Raman
    fr = 0.18

    num_modes = 10
    total_energy = 600.0 # nJ
    
    Nt = 2**12
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
    # calculate corresponding lambda
    wvls = 2 * np.pi * c0 / omega

    lambda0_nm = 1030.0

    omega0 = 2 * np.pi * c0 / (lambda0_nm * 1e-9)  # rad/s
    omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # rad/s
    omega_abs = omega0 + omega_rel

    wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

    coeffs = 0.01 * torch.ones(num_modes, dtype=torch.float32, device=device)
    coeffs[0] = 1.0
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs)**2))

    if DATA_FROM_MATLAB:
        from scipy.io import loadmat
        betas = loadmat('./data/betas_10modes.mat')['betas']
        betas = np.transpose(betas)  # (P, K)
        unit_conversion = 0.001 ** ((np.arange(-1, betas.shape[1]-1)))  # column vector
        unit_conversion = unit_conversion.reshape(1, -1)
        betas = betas * unit_conversion  # (P, K) -> (P, K) with unit conversion
        
        S = loadmat('./data/S_tensors_10modes.mat')['SR']
        print(f'betas shape: {betas.shape}, S shape: {S.shape}')

        hrw = loadmat('./data/hrw.mat')['hrw'].squeeze()

        #convert to torch
        betas = torch.tensor(betas, dtype=torch.float32, device=device)
        S = torch.tensor(S, dtype=torch.complex64, device=device)
        hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)
    else:
        raise NotImplementedError("Data loading from MATLAB is not implemented. Set DATA_FROM_MATLAB to True to load data.")

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian', values=None)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, num_save=100, dispersion=DISPERSION, nonlinear=NONLINEAR, raman=RAMAN, self_steeping=SELF_STEEPING)
    config_no_raman = SimConfig(center_wavelength=wvl0, num_save=100, dispersion=DISPERSION, nonlinear=NONLINEAR, raman=False, self_steeping=SELF_STEEPING)
    # Define simulation
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    sim_no_raman = Simulation(domain, fiber, initial_fields, boundary, config_no_raman)
    

    if DO_OPTIMIZE:
        # ---- 1300 nm에 가장 가까운 bin index (시작 시 1회 계산) ----
        idx_1300 = int(np.argmin(np.abs(wvls_nm - 1300.0)))

        # ---- 학습 변수: theta (실수), softplus→정규화→coeffs (in-phase) ----
        theta = torch.rand(num_modes, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([theta], lr=1e-2)

        total_iters = 1  # 필요에 맞게

        grad_fd = torch.zeros(num_modes, dtype=torch.float32, device=device)
        loss_fd = torch.zeros(num_modes, dtype=torch.float32, device=device)

        eps = 1e-8

        optimizer.zero_grad()
        
        def theta_to_coeffs(theta):
            a = torch.nn.functional.relu(theta) + 1e-12  # 양수 보장 + underflow 보호
            return a

        for it in range(total_iters):
            coeffs = theta_to_coeffs(theta)

            initial_fields = Pulse(
                domain, coeffs, tfwhm=tfwhm, total_energy=total_energy,
                p=1, C=0, t_center=0, type='gaussian', values=None
            )

            sim = Simulation(domain, fiber, initial_fields, boundary, config)
            sim.run()

            output_fields = sim.output_fields  # (P, Nt) complex, 주파수영역
            loss = objective_function(output_fields, idx_1300)

            optimizer.zero_grad()
            loss.backward()

            grad_ad = theta.grad.detach()
            loss_base = loss.detach()

            # finite difference check
            with torch.no_grad():
                for i in range(num_modes):
                    # x+eps
                    tp = theta.detach().clone()
                    tp[i] += eps
                    coeffs_p = theta_to_coeffs(tp)  # (P,) real
                    sim_p = Simulation(domain, fiber, Pulse(domain, coeffs_p, tfwhm=tfwhm,
                                    total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian'),
                                    boundary, config)
                    sim_p.run()
                    f_p = objective_function(sim_p.output_fields, idx_1300)

                    # x-eps
                    tm = theta.detach().clone()
                    tm[i] -= eps
                    coeffs_m = theta_to_coeffs(tm)

                    sim_m = Simulation(domain, fiber, Pulse(domain, coeffs_m, tfwhm=tfwhm,
                                    total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian'),
                                    boundary, config)
                    sim_m.run()
                    f_m = objective_function(sim_m.output_fields, idx_1300)

                    grad_fd[i] = (f_p - f_m) / (2*eps)

            ok = torch.allclose(grad_ad.cpu(), grad_fd.cpu(), atol=1e-3, rtol=1e-2)
            print("Grad check central:", ok)
            print("AD:", grad_ad)
            print("FD:", grad_fd) 

            optimizer.step()
            print(f"[{it:03d}] J(peak@1300nm) = {loss.item():.6e}")


        # 최종 최적 coeffs를 보존(원한다면 이후 재사용)
        coeffs = coeffs.detach()
        final_coeffs = coeffs.cpu().numpy()
        print(f'Final coefficients: {final_coeffs}')
    else:
        # Just run the simulation without optimization
        sim.run()

    output_fields = sim.output_fields.detach().cpu().numpy()
    output_intensity = np.abs(output_fields)**2


    num_modes = 10
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axes[0]
    ax1.set_title('Initial Pulse', fontsize=16)
    initial_fields_np = initial_fields.fields.detach().cpu().numpy() # PyTorch 텐서일 경우를 대비
    for i in range(num_modes):
        ax1.plot(domain.t, np.abs(initial_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8,  linewidth=2.0)
    
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Time (ps)', fontsize=14)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=14)


    ax2 = axes[1]
    ax2.set_title('Output Pulse', fontsize=16)
    for i in range(num_modes):
        # ax2.plot(t, output_intensity[i], '-', label=f'mode {i}', alpha=0.8, linewidth=2.0)
        ax2.plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    
    ax2.legend(fontsize=14)
    ax2.set_xlim([-1, 1])
    ax2.set_xlabel('Time (ps)', fontsize=14)

    ax3 = axes[2]
    ax3.set_title('Output Spectrum', fontsize=16)
    for i in range(num_modes):
        spectrum = np.abs(np.fft.fft(np.fft.ifftshift(output_fields[i])))**2
        idx = np.argsort(wvls_nm)
        wvls_nm = wvls_nm[idx]
        spectrum = spectrum[idx]
        ax3.plot(wvls_nm, spectrum, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    ax3.legend(fontsize=14)
    ax3.set_xlabel('Wavelength (nm)', fontsize=14)

    plt.tight_layout()
    plt.show()
