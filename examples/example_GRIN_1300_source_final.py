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


def objective_function(output_fields, target_ind):
    """
    Objective function to maximize the peak intensity of the fundamental modeat 1300 nm.
    """
    J = torch.sum(torch.abs(output_fields[0, target_ind])**2) 
    return -J  # We minimize the negative to maximize the peak intensity

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
    # calculate corresponding lambda
    wvls = 2 * np.pi * c0 / omega


    lambda0_nm = 1030.0 # nm

    omega0 = 2 * np.pi * c0 / (lambda0_nm * 1e-9)  # rad/s
    omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # rad/s
    omega_abs = omega0 + omega_rel

    wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

    coeffs = 0.01 * torch.ones(num_modes, dtype=torch.float32, device=device)
    coeffs[0] = 1.0
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs)**2))


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
    
    # We will use the predefined fields
    import scipy.io as sio
    initdata = sio.loadmat('./data/GRIN_1030_SPMGRINA_single_gpu_mpa_spm.mat')
    custom_fields = initdata['prop_output']['fields'][0,0][:,:,0]
    # transpose custom_fields
    custom_fields = np.transpose(custom_fields)
    custom_fields = torch.tensor(custom_fields, dtype=torch.complex64, device=device)


    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='custom', values=custom_fields)
    boundary = Boundary('periodic')
    config = SimConfig(center_wavelength=wvl0, dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING)    # Define simulation
    sim = Simulation(domain, fiber, initial_fields, boundary, config)
    
    field_initial = initial_fields.fields

    if DO_OPTIMIZE:
        idx_1300 = int(np.argmin(np.abs(wvls_nm - 1300.0)))

        theta = torch.rand(num_modes, dtype=torch.float32, device=device, requires_grad=True)
        optimizer = torch.optim.SGD([theta], lr=1e-2)

        total_iters = 1 

        grad_fd = torch.zeros(num_modes, dtype=torch.float32, device=device)
        loss_fd = torch.zeros(num_modes, dtype=torch.float32, device=device)

        eps = 1e-8

        optimizer.zero_grad()
        
        def theta_to_coeffs(theta):
            # theta: (P,) real (requires_grad=True)
            # p = torch.nn.functional.softmax(theta, dim=0) + 1e-12   # 안정성용 작은 값
            # a = torch.sqrt(p)
            # a = theta**2                                       # L2=1 자동
            a = torch.nn.functional.relu(theta) + 1e-12  # 양수 보장 + underflow 보호
            # a = a / torch.linalg.vector_norm(a)
            return a

        for it in range(total_iters):
            # a_raw = torch.nn.functional.softplus(theta) + 1e-12
            # a = a_raw / torch.linalg.vector_norm(a_raw)              # (P,)
            # coeffs_norm = a.to(torch.complex128)                     # 복소 계수(허수 0)

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
                    tp = theta.detach().clone()
                    tp[i] += eps
                    coeffs_p = theta_to_coeffs(tp)  # (P,) real
                    # a_raw_p = torch.nn.functional.softplus(tp) + 1e-12
                    # a_p = a_raw_p / torch.linalg.vector_norm(a_raw_p)
                    # coeffs_p = a_p.to(torch.complex128)
                    sim_p = Simulation(domain, fiber, Pulse(domain, coeffs_p, tfwhm=tfwhm,
                                    total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian'),
                                    boundary, config)
                    sim_p.run()
                    f_p = objective_function(sim_p.output_fields, idx_1300)

                    # x-eps
                    tm = theta.detach().clone()
                    tm[i] -= eps
                    coeffs_m = theta_to_coeffs(tm)
                    # a_raw_m = torch.nn.functional.softplus(tm) + 1e-12
                    # a_m = a_raw_m / torch.linalg.vector_norm(a_raw_m)
                    # coeffs_m = a_m.to(torch.complex128)
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
        # save the saved fields
        np.save('./data/saved_fields.npy', sim.saved_fields)

    output_fields = sim.output_fields.detach().cpu().numpy()
    output_intensity = np.abs(output_fields)**2
    
    # Input comparison with MATLAB
    if COMP_MATLAB:
        from scipy.io import loadmat
        # load input and output from MATLAB files
        initial_fields_gt = loadmat('./data/initial_field_gt.mat')['initial_fields']
        output_fields_gt = loadmat('./data/GRIN_1030_SPMGRINANSS_raman.mat')['prop_output']['fields'][0][0][:,:,-1]
        output_intensities_gt = np.abs(output_fields_gt)**2
        

    num_modes = 10
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax1 = axes[0]
    ax1.set_title('Initial Pulse', fontsize=16)
    initial_fields_np = initial_fields.fields.detach().cpu().numpy() # PyTorch 텐서일 경우를 대비
    for i in range(num_modes):
        ax1.plot(domain.t, np.abs(initial_fields_np[i])**2, '-', label=f'mode {i+1}', alpha=0.8,  linewidth=2.0)
    # xlim from -1 to 1
    if COMP_MATLAB:
        ax1.plot(domain.t, np.abs(initial_fields_gt[:,0])**2, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
    ax1.legend(fontsize=14)
    ax1.set_xlabel('Time (ps)', fontsize=14)
    ax1.set_ylabel('Intensity (a.u.)', fontsize=14)

    ax2 = axes[1]
    ax2.set_title('Output Pulse', fontsize=16)
    for i in range(num_modes):
        # ax2.plot(t, output_intensity[i], '-', label=f'mode {i}', alpha=0.8, linewidth=2.0)
        ax2.plot(domain.t, np.abs(output_fields[i]), '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    if COMP_MATLAB:
        # ax2.plot(t, output_intensities_gt[:, 0], 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
        ax2.plot(domain.t, np.abs(output_fields_gt[:,0]), 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
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

    if COMP_MATLAB:
        spectrum_gt = np.abs(np.fft.fft(np.fft.ifftshift(output_fields_gt[:,0])))**2
        spectrum_gt = spectrum_gt[idx]
        ax3.plot(wvls_nm, spectrum_gt, 'k--', label='groundtruth', alpha=0.8, linewidth=1.5)
    ax3.legend(fontsize=14)
    ax3.set_xlabel('Wavelength (nm)', fontsize=14)

    plt.tight_layout()
    plt.show()
