import torch
import math, os
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm
is_slurm_job = 'SLURM_JOB_ID' in os.environ
C0 = 299792458  # m/s

@dataclass
class SimConfig:
    center_wavelength: float
    num_save: int = -1
    dispersion: bool = True
    kerr: bool = True
    raman: bool = True
    self_steeping: bool = True
    num_chunks: int = 1

class Simulation:
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

    # --- self-steepening ---
    def _shock_correction(self, N_t, gamma, omega0):
        """
        N_t : (P, Nt) complex  (비선형 소스)
        반환 : -(gamma/omega0) * dN/dt  (P, Nt) complex
        """
        dev = N_t.device
        # ∂t N = IFFT{ iΩ * FFT(N) }
        Om = self.domain.omega.to(device=dev, dtype=torch.float64)  # (Nt,)
        dN_dt = torch.fft.ifft(1j * Om * torch.fft.fft(N_t, dim=-1), dim=-1)
        return -(gamma / omega0) * dN_dt
    
    # --- linear operator D[p, ω] ---
    def _build_linear_operator(self):
        P, Nt = self.num_modes, self.domain.Nt
        device = self.fields.fields.device

        omega = self.domain.omega.to(device=device, dtype=torch.float32)

        beta0_ref = float(torch.as_tensor(self.fiber.betas[0][0]).real)
        beta1_ref = float(torch.as_tensor(self.fiber.betas[0][1]).real)

        D = torch.zeros((P, Nt), dtype=torch.complex64, device=device)

        for p in range(P):
            betap = [float(b) for b in self.fiber.betas[p]]
            # k=0,1 terms
            poly = torch.zeros(Nt, dtype=torch.float32, device=device)
            poly = poly + (betap[0] - beta0_ref)                   
            poly = poly + (betap[1] - beta1_ref) * omega
            # k≥2 terms
            for k in range(2, len(betap)):
                poly = poly + betap[k] * (omega ** k) / math.factorial(k)
            D[p] = poly.to(torch.complex64)

        return D

    # --- dA/dz (instant part) ---
    def _kerr(self, A_t, gamma):
        SK = self.fiber.S
        S_kerr = torch.einsum('pqrs,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())
        dA = 1j * gamma * S_kerr
        return dA
    

    def _raman(self, A_t, gamma, fr, hrw,):
        pass

    # --- dA/dz (Raman part) ---
    def _build_nonlinear_operator(self, A_t, gamma, fr, hrw,):
        """
        A_t   : (P,Nt) complex, time domain
        gamma : scalar (W^-1 m^-1) 계수
        fr    : Raman fraction (0~1)
        hrw   : (Nt,) Raman response H_R(Ω)
        """
         # (1) instant part (1-fr)
        # SK = self.fiber.S  # (P,P,P,P)
        # S_kerr = torch.einsum('pqrs,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())
        # dA_kerr_t = 1j * gamma * (1.0 - fr) * S_kerr

        if self.config.kerr:
            dA_kerr_t = self._kerr(A_t, gamma)
        else:
            dA_kerr_t = 0.0


        # (2) Raman delayed part
        T = torch.einsum('mt,nt->mnt', A_t, A_t.conj())  # (P, P, Nt)

        # Convolve T with Raman response in time (via FFT)
        T_w = torch.fft.fft(T, dim=-1)  # (P, P, Nt)
        hrw_broadcast = hrw.conj().view(1, 1, -1)  # (1, 1, Nt)
        T_conv = torch.fft.ifft(T_w * hrw_broadcast, dim=-1)  # (P, P, Nt)
        T_conv = T_conv * self.domain.dt  # time step normalization

        Vpl = torch.einsum('pmrs,rst->pmt', self.fiber.S, T_conv)  # (P, P, Nt)

        S_raman = torch.einsum('pmt,mt->pt', Vpl, A_t)  # (P, Nt)
        dA_raman_t = 1j * gamma * S_raman

        # S_total = (1.0 - fr) * S_kerr + fr * S_raman
        dA = (1.0 - fr) * dA_kerr_t + fr * dA_raman_t

        sw = 1.0
        if self.config.self_steeping:
            omega0 = 2.0 * math.pi * C0 / float(self.fiber.wvl0) * 1e-12
            Omega = self.domain.omega.to(device=A_t.device, dtype=torch.float32).view(1, -1)  
          
            prefactor = (1j * gamma) * (1 + sw * (Omega / omega0))
            dA = torch.fft.fft(prefactor * torch.fft.ifft(dA, dim=-1) , dim=-1).to(A_t.dtype)
    
        return dA 

    def _propagate_one_step(self, fields, is_save_fields=False):
        if self.config.dispersion: 
            fields = fields * torch.exp(1j * self.domain.dz / 2 * self.D)

        fields = torch.fft.fft(fields, dim=-1)  

        # if is_save_fields:
        #     self.saved_fields[self.cnt] = fields
        #     self.cnt += 1

        k1 = self.domain.dz * self._build_nonlinear_operator(fields, self.gamma, self.fiber.fr, self.fiber.hrw,)
        k2 = self.domain.dz * self._build_nonlinear_operator(fields + 0.5 * k1, self.gamma, self.fiber.fr, self.fiber.hrw, )
        k3 = self.domain.dz * self._build_nonlinear_operator(fields + 0.5 * k2, self.gamma, self.fiber.fr, self.fiber.hrw, )
        k4 = self.domain.dz * self._build_nonlinear_operator(fields + k3, self.gamma, self.fiber.fr, self.fiber.hrw,)

        fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        if is_save_fields:
            self.saved_fields[self.cnt] = fields
            self.cnt += 1

        fields = torch.fft.ifft(fields, dim=-1)
        if self.config.dispersion:
            fields = fields * torch.exp(1j * self.domain.dz / 2 * self.D)

        return fields

    def run(self, requires_grad=False, use_cp=False):

        if self.config.num_save > 0:
            self.saved_fields = torch.zeros(self.config.num_save+1, self.num_modes, self.domain.Nt, dtype=torch.complex64, device=self.fields.fields.device)
            self.cnt = 0
            self.saved_fields[self.cnt] = self.fields.fields

        with torch.set_grad_enabled(requires_grad):
            
            fields = torch.fft.ifft(self.fields.fields, dim=-1)
            save_interval = self.domain.Nz // self.config.num_save
            for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):
                if i % save_interval == 0:
                    is_save_fields = True
        
                if use_cp:
                    pass
                else:
                    fields = self._propagate_one_step(fields, is_save_fields)
                

            self.fields.fields = torch.fft.fft(fields, dim=-1)