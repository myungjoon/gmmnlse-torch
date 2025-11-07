import torch
import numpy as np
import math, os
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm
is_slurm_job = 'SLURM_JOB_ID' in os.environ
C0 = 299792458  # m/s

@dataclass
class SimConfig:
    center_wavelength: float
    num_save: int = 50
    dispersion: bool = True
    kerr: bool = True
    raman: bool = False
    self_steeping: bool = False
    use_checkpointing: bool = True  # New parameter to enable/disable checkpointing
    checkpoint_segments: int = 10   # Number of segments to checkpoint

class SimulationCheckpointed:
    def __init__(self, domain, fiber, fields, boundary, config=None):
        self.domain = domain
        self.fields = fields
        self.fiber = fiber
        self.boundary = boundary
        self.config = config
        self.num_modes = fields.fields.shape[0]

    def _build_linear_operator(self):
        P, Nt = self.num_modes, self.domain.Nt
        device = self.fields.fields.device

        omega = self.domain.omega.to(device=device, dtype=torch.float32)

        beta0_ref = float(torch.as_tensor(self.fiber.betas[0][0]).real)
        beta1_ref = float(torch.as_tensor(self.fiber.betas[0][1]).real)

        D = torch.zeros((P, Nt), dtype=torch.complex64, device=device)

        for p in range(P):
            betap = [float(b) for b in self.fiber.betas[p]]  # 길이 ≥ 2 가정
            # k=0,1 항: 기준 모드 값 빼기
            poly = torch.zeros(Nt, dtype=torch.float32, device=device)
            poly = poly + (betap[0] - beta0_ref)                   # β0 diff
            poly = poly + (betap[1] - beta1_ref) * omega           # β1 diff * Ω
            # k≥2 항
            for k in range(2, len(betap)):
                poly = poly + betap[k] * (omega ** k) / math.factorial(k)
            D[p] = poly.to(torch.complex64)

        return D

    def _dA_dz_instant_kerr(self, A_t, gamma):
        SK = self.fiber.S
        F_inst = torch.einsum('pqrs,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())
        dA = 1j * gamma * F_inst
        return dA
    

    def _dA_dz_kerr_raman(self, A_t, gamma, fr, hrw,):
        """
        A_t   : (P,Nt) complex, time domain
        gamma : scalar (W^-1 m^-1) 계수
        fr    : Raman fraction (0~1)
        hrw   : (Nt,) Raman response H_R(Ω)
        """
         # (1) instant part (1-fr)
        SK = self.fiber.S  # (P,P,P,P)
        S_kerr = torch.einsum('pqrs,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())

        dA_kerr_t = 1j * gamma * (1.0 - fr) * S_kerr

        # (2) Raman delayed part
        # Raman term for GNLSE (multimode): delayed nonlinear response
        # T[m3, m4, t] = A_{m3}(t) * conj(A_{m4}(t))
        T = torch.einsum('mt,nt->mnt', A_t, A_t.conj())  # (P, P, Nt)

        # Convolve T with Raman response in time (via FFT)
        # (h_R * T)(t) = IFFT( HR(Ω) * FFT(T) )
        T_w = torch.fft.fft(T, dim=-1)  # (P, P, Nt)
        hrw_broadcast = hrw.conj().view(1, 1, -1)  # (1, 1, Nt)
        T_conv = torch.fft.ifft(T_w * hrw_broadcast, dim=-1)  # (P, P, Nt)
        T_conv = T_conv * self.domain.dt  # time step normalization

        Vpl = torch.einsum('pmrs,rst->pmt', self.fiber.S, T_conv)  # (P, P, Nt)

        S_raman = torch.einsum('pmt,mt->pt', Vpl, A_t)  # (P, Nt)
        dA_raman_t = 1j * gamma * fr * S_raman

        S_total = (1.0 - fr) * S_kerr + fr * S_raman
        dA = dA_kerr_t + dA_raman_t

        sw = 1.0
        if self.config.self_steeping:
            omega0 = 2.0 * math.pi * C0 / float(self.fiber.wvl0) * 1e-12  # [rad/s]
            Omega = self.domain.omega.to(device=A_t.device, dtype=torch.float32).view(1, -1)  
          
            prefactor = (1j * gamma) * (1 + sw * (Omega / omega0))           # (1,Nt), complex
            dA = torch.fft.fft(prefactor * torch.fft.ifft(S_total, dim=-1) , dim=-1).to(A_t.dtype)

    
        return dA 

    def _simulation_step(self, fields, fields_w, D, gamma, i):
        """Single simulation step that can be checkpointed"""
        if self.config.dispersion:  # Half-step linear dispersion
            fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * D)
            fields = torch.fft.fft(fields_w, dim=-1)
        fields = torch.fft.fft(fields_w, dim=-1)  
        
        if self.config.kerr:
            if self.config.raman:
                k1 = self.domain.dz * self._dA_dz_kerr_raman(fields, gamma, self.fiber.fr, self.fiber.hrw,)
                k2 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k1, gamma, self.fiber.fr, self.fiber.hrw, )
                k3 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k2, gamma, self.fiber.fr, self.fiber.hrw, )
                k4 = self.domain.dz * self._dA_dz_kerr_raman(fields + k3, gamma, self.fiber.fr, self.fiber.hrw,)
            else:
                k1 = self.domain.dz * self._dA_dz_instant_kerr(fields, gamma)
                k2 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k1, gamma)
                k3 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k2, gamma)
                k4 = self.domain.dz * self._dA_dz_instant_kerr(fields + k3, gamma)
            
            fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        if self.config.dispersion:
            fields_w = torch.fft.ifft(fields, dim=-1)
            fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * D)

        return fields, fields_w

    def _simulation_segment(self, fields, fields_w, D, gamma, segment_size):
        """Simulate a segment of steps that can be checkpointed"""
        for i in range(segment_size):
            fields, fields_w = self._simulation_step(fields, fields_w, D, gamma, i)
        return fields, fields_w

    def run(self):
        print("Checkpointed simulation started")

        gamma = self.fiber.n2 * 2.0 * math.pi / self.fiber.wvl0
        D = self._build_linear_operator()

        save_freq = self.domain.Nz // self.config.num_save
        self.saved_fields = torch.zeros((self.config.num_save+1, self.num_modes, self.domain.Nt), dtype=torch.complex64, device=self.fields.fields.device)

        fields = self.fields.fields
        fields_w = torch.fft.ifft(fields, dim=-1)

        if self.config.use_checkpointing:
            # Use checkpointing for memory efficiency
            segment_size = self.domain.Nz // self.config.checkpoint_segments
            remaining_steps = self.domain.Nz
            
            for segment in tqdm(range(self.config.checkpoint_segments), disable=is_slurm_job):
                current_segment_size = min(segment_size, remaining_steps)
                current_step = segment * segment_size
                
                # Save fields at the beginning of each segment
                if current_step % save_freq == 0:
                    self.saved_fields[current_step // save_freq] = fields
                
                # Use checkpoint for this segment
                fields, fields_w = checkpoint(
                    self._simulation_segment, 
                    fields, 
                    fields_w, 
                    D, 
                    gamma, 
                    current_segment_size
                )
                remaining_steps -= current_segment_size
                
                if remaining_steps <= 0:
                    break
        else:
            # Original non-checkpointed version
            for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):
                if i % save_freq == 0:
                    self.saved_fields[i // save_freq] = fields

                fields, fields_w = self._simulation_step(fields, fields_w, D, gamma, i)

        if self.config.dispersion:    
            fields = torch.fft.fft(fields_w, dim=-1)
        self.saved_fields[-1] = fields  
        self.output_fields = fields
