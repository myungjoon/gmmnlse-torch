import torch
import math, os
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint
import numpy as np

from tqdm import tqdm
is_slurm_job = 'SLURM_JOB_ID' in os.environ
is_slurm_job = False
C0 = 299792458  # m/s

def create_damped_freq_window(Nt, device=None, dtype=torch.float64):
      """Mirror of create_damped_freq_window.m — asymmetric super-Gaussian
      damping only the positive-Ω (short-wavelength) edge."""
      f      = np.fft.ifftshift(np.arange(1, Nt + 1))
      fc     = Nt // 2 + 1                  # MATLAB floor(Nt/2)+1
      ffwhm  = Nt * 0.85
      f0     = ffwhm / (2.0 * np.sqrt(np.log(2.0)))
      gexpo  = 40
      W = np.exp(-(f - fc) ** gexpo / (2.0 * f0 ** gexpo)) ** 20
      W[Nt // 2:] = 1.0                     # MATLAB W(fc:end)=1 → Python [Nt//2:]
      return torch.tensor(W, dtype=dtype, device=device)


@dataclass
class SimConfig:
    num_save: int = -1
    dispersion: bool = True
    kerr: bool = True
    raman: bool = True
    self_steeping: bool = True
    gain: bool = False
    loss: bool = False
    num_chunks: int = 1

class Simulation:
    """
    RK4IP (fixed step) for the GMMNLSE.
    Field is kept in the FREQUENCY domain between steps, matching MATLAB's
    stepping_RK4IP_nogain_adaptive.m. Convention:
        ifft = time -> freq    (physics, +iωt)
        fft  = freq -> time
    """

    def __init__(self, domain, fiber, fields, boundary, config=None):
        self.domain = domain
        self.fields = fields
        self.fiber = fiber
        self.boundary = boundary
        self.config = config
        self.num_modes = fields.fields.shape[0]
        self.gamma = self.fiber.n2 * 2.0 * math.pi / self.fiber.wvl0
        self.D = self._build_linear_operator()
        self.cnt = 0

        # Precompute things that don't change with z
        self._expD = torch.exp(1j * self.domain.dz / 2 * self.D)        # (P, Nt)
        omega0 = 2.0 * math.pi * C0 / float(self.fiber.wvl0) * 1e-12     # rad/ps
        Omega = self.domain.omega.to(device=self.D.device, dtype=torch.float64).view(1, -1)
        sw = 1.0 if self.config.self_steeping else 0.0
        self._prefactor = (1j * self.gamma) * (1.0 + sw * (Omega / omega0))   # (1, Nt)
        self._damped_window = create_damped_freq_window(                                                                                         
            self.domain.Nt, device=self.D.device, dtype=torch.float64
        ).view(1, -1)
        self._prefactor = self._prefactor * self._damped_window


    # --- linear operator D[p, ω] (no 1j; that goes in the exp) ---
    def _build_linear_operator(self):
        P, Nt = self.num_modes, self.domain.Nt
        device = self.fields.fields.device

        omega = self.domain.omega.to(device=device, dtype=torch.float64)

        beta0_ref = float(torch.as_tensor(self.fiber.betas[0][0]).real)
        beta1_ref = float(torch.as_tensor(self.fiber.betas[0][1]).real)

        D = torch.zeros((P, Nt), dtype=torch.complex128, device=device)

        for p in range(P):
            betap = [float(b) for b in self.fiber.betas[p]]
            poly = torch.zeros(Nt, dtype=torch.float64, device=device)
            poly = poly + (betap[0] - beta0_ref)
            poly = poly + (betap[1] - beta1_ref) * omega
            for k in range(2, len(betap)):
                poly = poly + betap[k] * (omega ** k) / math.factorial(k)
            D[p] = poly.to(torch.complex128)

        return D

    # --- bare Kerr sum (no 1j*gamma) ---
    def _kerr(self, A_t):
        SK = self.fiber.S
        return torch.einsum('srqp,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())

    # --- bare Raman sum (no 1j*gamma) ---
    def _raman_sum(self, A_t):
        hrw = self.fiber.hrw
        T = torch.einsum('mt,nt->mnt', A_t, A_t.conj())              # (P, P, Nt)
        T_w = torch.fft.fft(T, dim=-1)
        hrw_broadcast = hrw.conj().view(1, 1, -1)
        T_conv = torch.fft.ifft(T_w * hrw_broadcast, dim=-1) * self.domain.dt
        Vpl = torch.einsum('srmp,rst->pmt', self.fiber.S, T_conv)     # (P, P, Nt)
        return torch.einsum('pmt,mt->pt', Vpl, A_t)                   # (P, Nt)

    # --- N_op: freq-domain dA/dz, matches MATLAB N_op exactly ---
    def _N_op(self, A_w):
        """
        A_w : (P, Nt) complex, FREQUENCY domain
        returns dA/dz in FREQUENCY domain:
            Up   = (1 - fr)*Kerr + fr*Raman                       (time, bare)
            dAdz = n2_prefactor * ifft(Up)                        (freq)
        """
        A_t = torch.fft.fft(A_w, dim=-1)                              # freq -> time

        if self.config.kerr:
            S_kerr = self._kerr(A_t)
        else:
            S_kerr = torch.zeros_like(A_t)
        fr = float(self.fiber.fr)
        if self.config.raman:
            S_raman = self._raman_sum(A_t)
            Up = (1.0 - fr) * S_kerr + fr * S_raman   
        else:
            S_raman = torch.zeros_like(A_t)
            Up = S_kerr   

        
                            
        # Apply 1j*gamma*(1 + sw*Ω/ω0) once, in freq, then return in freq
        dA_w = self._prefactor * torch.fft.ifft(Up, dim=-1)           # time -> freq, with prefactor
        return dA_w.to(A_w.dtype)

    # --- one fixed-step RK4IP step (all in freq domain) ---
    def _propagate_one_step(self, A_w, is_save_fields=False):
        dz   = self.domain.dz
        expD = self._expD

        A_IP = expD * A_w

        # Equivalent with k_i = dz * a_i:
        k1 = dz * expD * self._N_op(A_w)
        k2 = dz *        self._N_op(A_IP + k1 / 2)
        k3 = dz *        self._N_op(A_IP + k2 / 2)
        k4 = dz *        self._N_op(expD * (A_IP + k3))

        A1w = expD * (A_IP + (k1 + 2 * k2 + 2 * k3) / 6) + k4 / 6

        if is_save_fields:
            self.saved_fields[self.cnt] = torch.fft.fft(A1w, dim=-1)  # freq -> time for storage
            self.cnt += 1

        return A1w

    def run(self, requires_grad=False, use_cp=False):

        if self.config.num_save > 0:
            self.saved_fields = torch.zeros(self.config.num_save + 1, self.num_modes, self.domain.Nt,
                                            dtype=torch.complex128, device=self.fields.fields.device)
            self.cnt = 0
            self.saved_fields[self.cnt] = self.fields.fields    # initial condition (time domain)
            self.cnt += 1

        with torch.set_grad_enabled(requires_grad):

            # time -> freq once, propagate in freq throughout
            A_w = torch.fft.ifft(self.fields.fields, dim=-1)
            save_interval = self.domain.Nz // self.config.num_save

            for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):
                is_save_fields = ((i + 1) % save_interval == 0)
                A_w = self._propagate_one_step(A_w, is_save_fields)

            # back to time domain for the final output
            self.fields.fields = torch.fft.fft(A_w, dim=-1)
