import argparse
import os
import time

import numpy as np
import torch

from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig
from gmmnlse import c0

DISPERSION = True
KERR = True
RAMAN = True
SELF_STEEPING = True

NUM_SAVE = 23

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_hrw(ts, Nt, t1=12.2e-3, t2=32e-3, tb=96e-3,
            fa=0.75, fb=0.21, fc=0.04):
    ha_model = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
    hb_model = ((2 * tb - ts) / tb**2) * np.exp(-ts / tb)
    ha = fa * ha_model + (fc * ha_model + fb * hb_model)   # scalar + linear pol
    hrw = np.fft.ifft(ha) * Nt
    return hrw

def sample_coeffs(rng, num_modes, random_phase=False, num_active=6):
    """Excite only the first num_active modes with equal amplitude; the rest are 0.
    If random_phase=True, only the phases are randomized in [0, 2π)."""
    num_active = min(num_active, num_modes)
    c = np.zeros(num_modes, dtype=np.complex128)
    if random_phase:
        phases = rng.uniform(0, 2 * np.pi, num_active)
        c[:num_active] = np.exp(1j * phases)
    else:
        c[:num_active] = 1.0
    c /= np.linalg.norm(c)
    return c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None,
                        help='If not given, a random integer is generated each run')
    parser.add_argument('--num_data', type=int, default=2000,
                        help='Number of random mode combinations (total samples = num_data * len(energy_list))')
    parser.add_argument('--num_modes', type=int, default=28)
    parser.add_argument('--energies', type=float, nargs='+',
                        default=[1, 5, 10, 15, 20, 25, 30], help='nJ')
    parser.add_argument('--random_phase', action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Randomize mode coefficient phases (default True; disable with --no-random_phase)')
    parser.add_argument('--out_dir', type=str, default='./data-measured')
    parser.add_argument('--fiber_path', type=str, default='./fibers/OM4-1030nm/')
    parser.add_argument('--num_active', type=int, default=6,
                        help='Number of excited modes (index 0 ~ num_active-1)')
    args = parser.parse_args()

    # If no seed is given, generate a random integer from OS entropy
    seed = args.seed
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'big')  # 0 ~ 4294967295

    num_modes = args.num_modes
    num_data = args.num_data
    energy_list = args.energies
    num_energy = len(energy_list)
    num_active = args.num_active
    num_total = num_data * num_energy

    os.makedirs(args.out_dir, exist_ok=True)

    # Independent RNG per seed (safe for running multiple processes in parallel)
    rng = np.random.default_rng(seed)

    print(f'seed: {seed}, num_modes: {num_modes}, num_data: {num_data}, '
          f'energies: {energy_list} nJ -> total {num_total} samples, device: {device}')

    # ---------------- Simulation parameters ----------------
    dz = 5e-5
    L0 = 0.23
    Nz = int(L0 / dz)

    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.245

    Nt = 2**9
    time_window = 10.1  # ps
    dt = time_window / Nt
    dt_s = dt * 1e-12
    tfwhm = 0.270  # ps
    t_center = 0.0

    ts = dt * np.arange(Nt)
    hrw = torch.tensor(get_hrw(ts, Nt), dtype=torch.complex128, device=device)

    # ---------------- Fiber data ----------------
    S = np.load(f'{args.fiber_path}S.npy')
    S = S[:num_modes, :num_modes, :num_modes, :num_modes]
    S = torch.tensor(S, dtype=torch.complex128, device=device)

    betas = np.load(f'{args.fiber_path}betas.npy')[:num_modes]
    betas = torch.tensor(betas, dtype=torch.float64, device=device)

    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw)

    start_time = time.time()
    sim_count = 0

    # ----- measured pulse (FROG-retrieved) -----
    pulse_data = np.load('input_pulse_amplified.npz')
    measured_time = pulse_data['time'] * 1e12   # s -> ps
    measured_field = pulse_data['pulse']        # complex envelope

    for n in range(num_data):
        tag = f'M{num_modes}_seed{seed}_n{n:05d}'
        set_path = os.path.join(args.out_dir, f'set_{tag}.npz')

        coeffs_np = sample_coeffs(rng, num_modes,
                                  random_phase=args.random_phase,
                                  num_active=args.num_active)
        actual_energies = np.array(energy_list) * (1 + 0.2 * (rng.random(num_energy) - 0.5))

        if os.path.exists(set_path):
            print(f'[{n + 1}/{num_data}] {set_path} exists, skipping')
            continue

        coeffs = torch.tensor(coeffs_np, dtype=torch.complex128, device=device)

        set_input = np.zeros((num_energy, num_modes, Nt), dtype=np.complex128)
        set_output = np.zeros((num_energy, num_modes, Nt), dtype=np.complex128)

        set_field = np.zeros((num_energy, (NUM_SAVE+1), num_modes, Nt), dtype=np.complex128)
        for i in range(len(energy_list)):

            pulse_energy = actual_energies[i]
            initial_fields = Pulse(domain, coeffs, pulse_energy=pulse_energy,
                           type='measured', measured_time=measured_time,
                           measured_field=measured_field, t_center=t_center)

            set_input[i] = initial_fields.fields.detach().cpu().numpy()

            boundary = Boundary('periodic')
            config = SimConfig(dispersion=DISPERSION, kerr=KERR, raman=RAMAN,
                               self_steeping=SELF_STEEPING, num_save=NUM_SAVE)

            sim = Simulation(domain, fiber, initial_fields, boundary, config)
            sim.run()

            set_output[i] = sim.fields.fields.detach().cpu().numpy()
            sim_count += 1

            set_field[i] = sim.saved_fields.detach().cpu().numpy()

            elapsed = time.time() - start_time
            print(f'[{n + 1}/{num_data}] E={pulse_energy} nJ '
                  f'elapsed: {elapsed:.1f} s '
                  f'(avg {elapsed / sim_count:.1f} s/sim)')

        # Write to a temp file, then rename (avoids corrupted files if the process dies mid-write)
        tmp_path = set_path + '.tmp.npz'
        np.savez(tmp_path,
                 input_fields=set_input,
                 output_fields=set_output,
                 fields=set_field,
                 coeffs=coeffs_np,
                 energies=actual_energies,              # actual energies used (nJ)
                 seed=seed, index=n)
        os.replace(tmp_path, set_path)
        print(f'  -> saved {set_path}')

    print(f'Done. {num_data} sets ({sim_count} sims) in {args.out_dir}')

if __name__ == '__main__':
    main()