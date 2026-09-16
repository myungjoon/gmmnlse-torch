import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import plot_temporal_evolution, plot_spectral_evolution
from gmmnlse.mode import ModeSolver
from gmmnlse import c0

import os, time

from pathlib import Path
from datetime import datetime

plt.rcParams['font.size'] = 15
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.tab20.colors)

def get_hrw(ts, t1=12.2e-3, t2=32e-3, tb=96e-3,
            fa=0.75, fb=0.21, fc=0.04):
    """Raman response in frequency domain (unnormalized IFFT, matches Fiber convention)."""
    ha = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
    hb = ((2 * tb - ts) / tb**2) * np.exp(-ts / tb)
    hr = (fa + fc) * ha + fb * hb          # scalar + linear-pol contributions
    return np.fft.ifft(hr, norm='forward')

def print_run_summary(*, fiber_path, num_modes, L, dz, Nz, Nt, time_window, dt,
                      tfwhm, energy_list, device):
    dt_fs = dt * 1e3
    flags = ' '.join(f'{k}={"on" if v else "off"}' for k, v in [
        ('dispersion', DISPERSION), ('kerr', KERR),
        ('raman', RAMAN), ('self_steepening', SELF_STEEPING)])
    print('=== gmmnlse run ===')
    print(f'fiber      : {fiber_path}  ({num_modes} modes)')
    print(f'L          : {L:.4f} m   dz : {dz:.1e} m   Nz : {Nz}')
    print(f'time grid  : Nt={Nt}   window={time_window:.1f} ps   dt={dt_fs:.2f} fs')
    print(f'pulse      : gaussian  tfwhm={tfwhm:.3f} ps')
    print(f'energies   : {energy_list} nJ')
    print(f'physics    : {flags}')
    print(f'save       : {NUM_SAVE_INTERVALS} intervals')
    print(f'device     : {device}')
    print('===================')

def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def load_fiber(fiber_path, num_modes, device):
    """Load S, betas, modes from fiber_path and validate consistency before use."""
    if not os.path.isdir(fiber_path):
        raise FileNotFoundError(f'fiber directory not found: {fiber_path}')

    arrs = {}
    for name in ('S', 'betas', 'modes'):
        p = os.path.join(fiber_path, f'{name}.npy')
        if not os.path.isfile(p):
            raise FileNotFoundError(f'missing fiber file: {p}')
        arrs[name] = np.load(p)

    n_avail = arrs['betas'].shape[0]
    if n_avail < num_modes:
        raise ValueError(f'requested {num_modes} modes but fiber has only {n_avail}')
    if arrs['S'].shape[:4] != (n_avail,) * 4 or arrs['modes'].shape[0] != n_avail:
        raise ValueError(f'inconsistent mode count: S {arrs["S"].shape}, '
                         f'betas {arrs["betas"].shape}, modes {arrs["modes"].shape}')

    S = torch.tensor(arrs['S'][:num_modes, :num_modes, :num_modes, :num_modes],
                     dtype=torch.complex128, device=device)
    betas = torch.tensor(arrs['betas'][:num_modes], dtype=torch.float64, device=device)
    modes = arrs['modes'][:num_modes]                                     # (P, Nx, Ny)
    modes = modes / np.sqrt(np.sum(modes**2, axis=(1, 2)))[:, None, None] # 정규화
    return S, betas, modes


def load_amplitudes(path, num_modes, device):
    """Load initial mode amplitudes and check they match num_modes."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'mode amplitude file not found: {path}')
    a = np.load(path)
    if a.ndim != 1 or a.shape[0] != num_modes:
        raise ValueError(f'amplitudes shape {a.shape} does not match num_modes={num_modes}')
    return torch.as_tensor(a, dtype=torch.complex128, device=device)


DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

NUM_SAVE_INTERVALS = 100


if __name__ == '__main__':
    
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    num_modes = 28    
    dz = 5e-5

    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.245

    energy_list = [1.6, 5, 10, 15, 20]

    Nt = 2**11
    time_window = 6.0 # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12  # s
    tfwhm = 0.300 # ps
    t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)
    t_center = 0.0

    freq = np.fft.fftfreq(Nt, dt_s)
    f0 = c0 / wvl0
    f = f0 + np.arange(-Nt//2, Nt//2) / (Nt * dt_s)
    wl = c0 / f * 1e9
    order = np.argsort(wl)
    wl_sorted = wl[order]

    ts = dt * np.arange(Nt)
    hrw = get_hrw(ts)
    hrw = torch.tensor(hrw, dtype=torch.complex128, device=device)

    fiber_path = './fibers/OM4-1030nm/'
    S, betas, modes = load_fiber(fiber_path, num_modes, device)

    amplitude_path = 'mode_amplitudes_1.6mW.npy'
    coeffs = load_amplitudes(amplitude_path, num_modes, device)

    L0 = 0.23
    L_tag = f'L{L0:.4f}'
    Nz = int(L0 / dz)

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw,)

    zero_coeffs = []

    num_energy = len(energy_list)
    all_input_fields  = np.zeros((num_energy, num_modes, Nt), dtype=np.complex128)
    all_output_fields = np.zeros((num_energy, num_modes, Nt), dtype=np.complex128)

    run_name = f'{datetime.now():%Y%m%d_%H%M%S}_{Path(fiber_path).name}_{num_modes}modes'
    out_dir = Path('results') / run_name
    fig_dir = out_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    peak_maps   = []
    peak_labels = []   
    temporal   = []
    peak_idx   = []

    for i,total_energy in enumerate(energy_list):
        initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=t_center, type='gaussian')
        input_fields = initial_fields.fields.detach().cpu().numpy()
        boundary = Boundary('periodic')
        config = SimConfig(dispersion=DISPERSION, kerr=KERR, raman=RAMAN, self_steeping=SELF_STEEPING, num_save=NUM_SAVE_INTERVALS)

        sync(device)
        t0 = time.perf_counter()
        sim = Simulation(domain, fiber, initial_fields, boundary, config)
        sim.run()
        sync(device)
        elapsed_time = time.perf_counter() - t0
        print(f'Total simulation time : {elapsed_time}')

        output_fields = sim.fields.fields.detach().cpu().numpy()
        total_fields = sim.saved_fields.detach().cpu().numpy()

        temporal.append(output_fields)

        all_input_fields[i]  = input_fields
        all_output_fields[i] = output_fields

        z = np.linspace(0, L0, NUM_SAVE_INTERVALS+1)
        intensity_t = np.abs(total_fields) ** 2

        # ---- zero delay 지점의 모드 projection (복소수) ----
        idx_t0 = int(np.argmin(np.abs(t - 0.0)))

        # 방법 1: 모드 계수 직접 추출 (직교정규 기저이므로 이것이 곧 projection)
        c_t0 = output_fields[:, idx_t0].copy()          # (num_modes,) complex

        # # 방법 2: 필드 합성 후 overlap integral로 재투영 (검증용)
        E_t0 = np.einsum('p,pxy->xy', c_t0, modes)         # (Nx, Ny) complex
        # c_t0_proj = np.einsum('xy,pxy->p', E_t0, modes)    # (num_modes,) complex

        # print(f'  E={total_energy} nJ, t={t[idx_t0]:.4f} ps, '
        #       f'projection err = {np.max(np.abs(c_t0 - c_t0_proj)):.3e}')

        zero_coeffs.append(c_t0)

        # spectrum
        spec = np.fft.fftshift(
        np.abs(np.fft.fft(np.fft.ifftshift(total_fields, axes=-1), axis=-1))**2,
        axes=-1,
        )

        fig, ax1 = plt.subplots(1, 2, figsize=(14, 6))

        for j in range(num_modes):
            line1, = ax1[0].plot(domain.t, np.abs(input_fields[j])**2, '-', label=f'mode {i+1}', alpha=0.8, linewidth=2.0)

        # ax1[0].legend(loc='upper left',fontsize=12)
        # ax1[0].set_xlim([-15, 15])
        ax1[0].set_xlabel('Time (ps)', fontsize=18)
        ax1[0].set_ylabel('Intensity (a.u.)', fontsize=18)
        ax1[0].legend(loc='upper right', fontsize=12)
        
        input_spectrum = np.fft.fftshift(
        np.abs(np.fft.ifft(np.fft.ifftshift(input_fields, axes=-1), axis=-1))**2,
        axes=-1,
        )
        for j in range(num_modes):
            ax1[1].plot(wl_sorted, input_spectrum[j][..., order], '-', label=f'mode {j+1}',)
        # ax2[1].set_xlim([700, 1400])
        # ax1[1].legend(loc='upper left', fontsize=12)
        ax1[1].set_xlabel('Wavelength (nm)', fontsize=18)
        ax1[1].set_ylabel('Intensity (a.u.)', fontsize=18)
        plt.savefig(f'fmf-input-{num_modes}-{total_energy}-2.png', dpi=300)
        plt.close(fig)

        # Plot current output and ground truth for each mode
        fig, ax2 = plt.subplots(1, 2, figsize=(14, 6))
        for j in range(num_modes):
            line1, = ax2[0].plot(domain.t, np.abs(output_fields[j])**2, '-', label=f'mode {j+1}', alpha=0.8, linewidth=2.0)
            # ax2.plot(domain.t, np.abs(output_fields_gt[i]), '--', alpha=0.8, linewidth=1.5, color=line1.get_color())

        # ax2[0].legend(loc='upper left', fontsize=12)
        # ax2[0].set_xlim([-1, 1])
        ax2[0].set_xlabel('Time (ps)', fontsize=18)
        ax2[0].set_ylabel('Intensity (a.u.)', fontsize=18)
        ax2[0].legend(loc='upper left', fontsize=12)

        output_spectrum = np.fft.fftshift(
        np.abs(np.fft.ifft(np.fft.ifftshift(output_fields, axes=-1), axis=-1))**2,
        axes=-1,
        )
        for j in range(num_modes):
            line1, = ax2[1].plot(wl_sorted, output_spectrum[j][...,order], '-', label=f'mode {j+1}', alpha=0.8, linewidth=2.0)
        ax2[1].legend(loc='upper left', fontsize=12)
        ax2[1].set_xlabel('Wavelength (nm)', fontsize=18)
        ax2[1].set_ylabel('Intensity (a.u.)', fontsize=18)
        ax2[1].set_xlim([930, 1130])
        
        plt.savefig(f'fmf-output-{num_modes}-{total_energy}-2.png', dpi=300)
        plt.close(fig)


        intensity_t_sum = intensity_t.sum(axis=1)      # (z, t)
        spec_sum = spec.sum(axis=1)[:, order]          # (z, wl) — order로 정렬

        # Time-resolved output intensity: 10 spatial snapshots from -0.5 ps to 0.5 ps

        snapshot_times = np.linspace(-0.5, 1.0, 6)
        snap_idx = [np.argmin(np.abs(t - tk)) for tk in snapshot_times]

        # I(x,y; t_k) = |sum_p A_p(t_k) psi_p(x,y)|^2
        A_snap = output_fields[:, snap_idx]                           # (P, 10)
        snap_maps = np.abs(np.einsum('pk,pxy->kxy', A_snap, modes))**2   # (10, Nx, Ny)

        vmax = snap_maps.max()
        fig, axes = plt.subplots(2, 3, figsize=(22, 9))
        for k, ax in enumerate(axes.flat):
            im = ax.imshow(snap_maps[k], cmap='turbo', vmin=0,)
            ax.set_title(f't = {t[snap_idx[k]]:.2f} ps', fontsize=20)
            ax.axis('off')
        # fig.colorbar(im, ax=axes, fraction=0.02, label='Intensity (a.u.)')
        plt.savefig(f'time_resolved_output-{num_modes}-{total_energy}-{dz}-2.png', dpi=300)
        plt.close(fig)


        # ---- 1) 모드 합성 전체 필드의 시간영역 intensity ----
        # E(x,y,t) = sum_p A_p(t) * psi_p(x,y)
        # I(t) = ∫|E|^2 dxdy = sum_p |A_p(t)|^2  (모드 직교정규 가정)
        I_t_total = np.sum(np.abs(output_fields) ** 2, axis=0)     # (Nt,)
        I_t_in    = np.sum(np.abs(input_fields) ** 2, axis=0)      # (Nt,)

        fig, ax = plt.subplots(figsize=(8, 5))
        # ax.plot(t, I_t_in / I_t_in.max(), '--', lw=2.0, label='input (all modes)')
        ax.plot(t, I_t_total / I_t_total.max(), '-', lw=2.0, label='output (all modes)')
        ax.set_xlabel('Time (ps)', fontsize=18)
        ax.set_ylabel('Intensity (a.u.)', fontsize=18)
        ax.set_xlim([-1.5, 1.5])
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'fullfield_temporal-{num_modes}-{total_energy}.png', dpi=300)
        plt.close(fig)

        # ---- 2) 중앙 구간 50개 샘플링 → 5 x 10 subplots ----
        t_half = 0.5                                   # 중앙 ±0.5 ps 구간 (필요시 조정)
        i0 = np.argmin(np.abs(t - (-t_half)))
        i1 = np.argmin(np.abs(t - (+t_half)))
        snap_idx = np.linspace(i0, i1, 50).astype(int)

        A_snap = output_fields[:, snap_idx]                              # (P, 50)
        E_snap = np.einsum('pk,pxy->kxy', A_snap, modes)                    # (50, Nx, Ny)
        I_snap = np.abs(E_snap) ** 2

        vmax = I_snap.max()
        fig, axes = plt.subplots(5, 10, figsize=(30, 15))
        for k, ax in enumerate(axes.flat):
            im = ax.imshow(I_snap[k], cmap='turbo', vmin=0, vmax=vmax)
            ax.set_title(f'{t[snap_idx[k]]:.3f} ps', fontsize=12)
            ax.axis('off')
        fig.colorbar(im, ax=axes, fraction=0.015, label='Intensity (a.u.)')
        plt.savefig(f'fullfield_snapshots-{num_modes}-{total_energy}.png', dpi=300)
        plt.close(fig)

        I_sum = I_snap.sum(axis=(1, 2))          # (50,)

        k_peak = int(np.argmax(I_sum))
        t_peak = t[snap_idx[k_peak]]

        peak_maps.append(I_snap[k_peak])
        peak_idx.append(snap_idx[k_peak])
        peak_labels.append((total_energy, t_peak))
    

        # ---- 전체 필드 스펙트럼 (모든 모드 합) ----
        in_spec_sum  = input_spectrum.sum(axis=0)[order]     # (Nt,)
        out_spec_sum = output_spectrum.sum(axis=0)[order]

        mm = (wl_sorted >= 930) & (wl_sorted <= 1130)

        fig, ax = plt.subplots(figsize=(8, 5))
        # ax.plot(wl_sorted[mm], in_spec_sum[mm]  / in_spec_sum[mm].max(),  '--', lw=2.0, label='input')
        ax.plot(wl_sorted[mm], out_spec_sum[mm] / out_spec_sum[mm].max(), '-',  lw=2.0, label='output')
        # ax.axvline(wvl0 * 1e9, color='k', ls=':', lw=1.2)
        ax.set_xlim([930, 1130])
        ax.set_xlabel('Wavelength (nm)', fontsize=18)
        ax.set_ylabel('Intensity (a.u.)', fontsize=18)
        ax.grid(alpha=0.3)
        # ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(f'fullfield_spectrum-{num_modes}-{total_energy}.png', dpi=300)
        plt.close(fig)

    ncols = len(energy_list)
    # fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 7.5))

    # --- 1행만: peak time intensity map ---
    fig, axes = plt.subplots(1, ncols, figsize=(2.8 * ncols, 3.2),
                             squeeze=False)
    for k in range(ncols):
        ax = axes[0, k]
        ax.imshow(peak_maps[k], cmap='turbo', vmin=0, vmax=peak_maps[k].max())

        ny, nx = peak_maps[k].shape          # (row, col) = (y, x)
        cx, cy = (nx - 1) / 2, (ny - 1) / 2  # 이미지 중앙 (픽셀 좌표)
        r = min(nx, ny) / 4                  # 반지름 = 전체 픽셀의 1/4
        ax.add_patch(Circle((cx, cy), r, fill=False,
                            edgecolor='white', linewidth=1.5, linestyle='-'))

        ax.axis('off')

    fig.suptitle('Peak-time intensity patterns', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f'fullfield_peak_temporal_all-{L_tag}.png', dpi=200)
    plt.close(fig)

    # # --- 2행: 모드별 시간영역 output ---
    # lines = []
    # for k in range(ncols):
    #     ax = axes[1, k]
    #     for m in range(num_modes):
    #         ln, = ax.plot(t, np.abs(temporal[k][m])**2, '-',
    #                     alpha=0.8, linewidth=1.5, label=f'mode {m+1}')
    #         if k == 0:
    #             lines.append(ln)               # 범례용 핸들은 첫 패널에서만 수집
    #     ax.set_xlim([-1.5, 1.5])
    #     ax.set_xlabel('Time (ps)', fontsize=14)
    #     if k == 0:
    #         ax.set_ylabel('Intensity (a.u.)', fontsize=14)

    # --- 공통 legend 하나만 ---
    # fig.legend(lines, [f'mode {m+1}' for m in range(num_modes)],
    #         loc='lower center', ncol=min(num_modes, 6), fontsize=12,
    #         frameon=False, bbox_to_anchor=(0.5, 0.0))

    # plt.tight_layout(rect=[0, 0.20, 1, 1])     # 아래쪽에 legend 공간 확보
    # plt.savefig(f'fullfield_peak_temporal_all-{L_tag}-{mode_tag}.png', dpi=300)
    # plt.close(fig)

    mode_frac = np.zeros((len(temporal), num_modes))
    mode_coeffs = np.zeros((len(temporal), num_modes), dtype=np.complex128)

    for k, A in enumerate(temporal):
        Ak = A[:, peak_idx[k]]
        P = np.abs(A[:, peak_idx[k]]) ** 2       # (P,) — 해당 시점의 모드별 power
        mode_frac[k] = P / P.sum()
        mode_coeffs[k] = Ak

    x = np.arange(num_modes)
    w = 0.8 / len(temporal)
    cmap = plt.get_cmap('viridis')

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k in range(len(temporal)):
        ax.bar(x + (k - (len(temporal) - 1) / 2) * w, mode_frac[k], w,
            label=f'{energy_list[k]} nJ', color=cmap(k /(len(temporal) - 1)))

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(num_modes)], fontsize=9)
    ax.set_xlabel('Mode index', fontsize=13)
    ax.set_ylabel('Mode content', fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f'mode_distribution_peak-{num_modes}.png', dpi=300)
    plt.close(fig)

    # --- phase bar graph: global phase = mode 1 @ lowest energy → 0 ---
    global_phase = np.exp(-1j * np.angle(mode_coeffs[0, 0]))

    fig, ax = plt.subplots(figsize=(10, 5))
    for k in range(len(temporal)):
        phases = np.angle(mode_coeffs[k] * global_phase)
        ax.bar(x + (k - (len(temporal) - 1) / 2) * w, phases, w,
            label=f'{energy_list[k]} nJ', color=cmap(k /(len(temporal) - 1)))

    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}' for i in range(num_modes)])
    ax.set_xlabel('Mode index', fontsize=18)
    ax.set_ylabel('Mode phase (rad)', fontsize=18)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_ylim([-np.pi, np.pi])
    ax.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    plt.savefig(f'mode_phase_peak-{num_modes}.png', dpi=300)
    plt.close(fig)

    np.save(f'input_fields_{num_modes}.npy', all_input_fields)
    np.save(f'output_fields_{num_modes}.npy', all_output_fields)
    np.save(f'energy_list_{num_modes}.npy',   np.array(energy_list))
    
    # zero_coeffs = np.array(zero_coeffs, dtype=np.complex128)   # (num_energy, num_modes)
    # np.save(f'zero_delay_coeffs_{num_modes}_{mode_tag}.npy', zero_coeffs)