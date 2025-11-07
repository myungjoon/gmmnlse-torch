import torch
import torch.nn as nn
import numpy as np
import math, os
from dataclasses import dataclass
from torch.utils.checkpoint import checkpoint

from tqdm import tqdm
is_slurm_job = 'SLURM_JOB_ID' in os.environ
C0 = 299792458  # m/s


def make_save_indices(Nz: int, n_save: int = 50):
    # 0..Nz 포함해서 균등 샘플 → 물리적으로는 step 완료 후 상태를 저장하는 걸 권장
    # (여기서는 마지막 z=Nz 포함)
    idx = torch.linspace(0, Nz, steps=n_save, dtype=torch.long)
    idx = torch.clamp(idx, 0, Nz)  # 안전
    # 중복 제거 & 정렬
    idx = torch.unique(idx, sorted=True)
    return idx.tolist()

@dataclass
class SimConfig:
    center_wavelength: float
    num_save: int = -1
    dispersion: bool = True
    kerr: bool = True
    raman: bool = False
    self_steeping: bool = False
    num_chunks: int = 1

class Simulation(nn.Module):
    def __init__(self, domain, fiber, fields, boundary, config=None):
        super(Simulation, self).__init__()
        self.domain = domain
        self.fields = fields
        self.fiber = fiber
        self.boundary = boundary
        self.config = config
        self.num_modes = fields.fields.shape[0]


        self.gamma = self.fiber.n2 * 2.0 * math.pi / self.fiber.wvl0
        self.D = self._build_linear_operator()

        if self.config.num_save == -1:
            self.config.num_save = 100
        self.save_freq = (self.domain.Nz + 1) // self.config.num_save

        if self.config.num_chunks <= 0:
            raise ValueError("num_chunks should be greater than 0.")

        self.checkpoint_interval = math.ceil(self.domain.Nz / self.config.num_chunks)
        # self.saved_fields = torch.zeros((self.config.num_save+1, self.num_modes, self.domain.Nt), dtype=torch.complex64, device=self.fields.fields.device)
            

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

    def run(self, initial_fields):
        # gamma = self.fiber.n2 * 2.0 * math.pi / self.fiber.wvl0
        # D = self._build_linear_operator()
        
        print("Simulation started") 

        fields = initial_fields
        fields_w = torch.fft.ifft(fields, dim=-1)
  
        for i in tqdm(range(self.domain.Nz), disable=is_slurm_job):
            # if i % self.save_freq == 0:
                # self.saved_fields[i // self.save_freq] = fields

            if self.config.dispersion:  # Half-step linear dispersion
                fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * self.D)
                fields = torch.fft.fft(fields_w, dim=-1)
            fields = torch.fft.fft(fields_w, dim=-1)  
            
            if self.config.kerr:
                if self.config.raman:
                    k1 = self.domain.dz * self._dA_dz_kerr_raman(fields, self.gamma, self.fiber.fr, self.fiber.hrw,)
                    k2 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k1, self.gamma, self.fiber.fr, self.fiber.hrw, )
                    k3 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k2, self.gamma, self.fiber.fr, self.fiber.hrw, )
                    k4 = self.domain.dz * self._dA_dz_kerr_raman(fields + k3, self.gamma, self.fiber.fr, self.fiber.hrw,)
                else:
                    k1 = self.domain.dz * self._dA_dz_instant_kerr(fields, self.gamma)
                    k2 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k1, self.gamma)
                    k3 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k2, self.gamma)
                    k4 = self.domain.dz * self._dA_dz_instant_kerr(fields + k3, self.gamma)
                
                fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6.0

            if self.config.dispersion:
                fields_w = torch.fft.ifft(fields, dim=-1)
                fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * self.D)

        if self.config.dispersion:    
            fields = torch.fft.fft(fields_w, dim=-1)
        # self.saved_fields[-1] = fields  
        self.fields.fields = fields

    def run_chunk(self, chunk_size):
        fields_w = torch.fft.ifft(self.fields.fields, dim=-1)
  
        for i in tqdm(range(chunk_size), disable=is_slurm_job):
            # if i % self.save_freq == 0:
                # self.saved_fields[i // self.save_freq] = self.fields.fields

            if self.config.dispersion:  # Half-step linear dispersion
                fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * self.D)
                fields = torch.fft.fft(fields_w, dim=-1)
            fields = torch.fft.fft(fields_w, dim=-1)  
            
            if self.config.kerr:
                if self.config.raman:
                    k1 = self.domain.dz * self._dA_dz_kerr_raman(fields, self.gamma, self.fiber.fr, self.fiber.hrw,)
                    k2 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k1, self.gamma, self.fiber.fr, self.fiber.hrw, )
                    k3 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k2, self.gamma, self.fiber.fr, self.fiber.hrw, )
                    k4 = self.domain.dz * self._dA_dz_kerr_raman(fields + k3, self.gamma, self.fiber.fr, self.fiber.hrw,)
                else:
                    k1 = self.domain.dz * self._dA_dz_instant_kerr(fields, self.gamma)
                    k2 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k1, self.gamma)
                    k3 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k2, self.gamma)
                    k4 = self.domain.dz * self._dA_dz_instant_kerr(fields + k3, self.gamma)
                
                fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6.0

            if self.config.dispersion:
                fields_w = torch.fft.ifft(fields, dim=-1)
                fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * self.D)

        if self.config.dispersion:    
            fields = torch.fft.fft(fields_w, dim=-1)

        # self.saved_fields[-1] = fields  
        self.fields.fields = fields
   
    def _one_step(self, fields, fields_w):
        # half-step linear
        if self.config.dispersion:
            fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * self.D)
            fields = torch.fft.fft(fields_w, dim=-1)

        # --- (여기서 Kerr/Raman 비선형) ---
        if self.config.kerr:
            if self.config.raman:
                k1 = self.domain.dz * self._dA_dz_kerr_raman(fields, self.gamma, self.fiber.fr, self.fiber.hrw)
                k2 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k1, self.gamma, self.fiber.fr, self.fiber.hrw)
                k3 = self.domain.dz * self._dA_dz_kerr_raman(fields + 0.5 * k2, self.gamma, self.fiber.fr, self.fiber.hrw)
                k4 = self.domain.dz * self._dA_dz_kerr_raman(fields + k3, self.gamma, self.fiber.fr, self.fiber.hrw)
            else:
                k1 = self.domain.dz * self._dA_dz_instant_kerr(fields, self.gamma)
                k2 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k1, self.gamma)
                k3 = self.domain.dz * self._dA_dz_instant_kerr(fields + 0.5 * k2, self.gamma)
                k4 = self.domain.dz * self._dA_dz_instant_kerr(fields + k3, self.gamma)
            fields = fields + (k1 + 2*k2 + 2*k3 + k4) / 6.0

        # half-step linear
        if self.config.dispersion:
            fields_w = torch.fft.ifft(fields, dim=-1)
            fields_w = fields_w * torch.exp(1j * self.domain.dz / 2 * self.D)

        return fields, fields_w

    def _run_steps_pure(self, fields_in, steps):
        fields = fields_in
        fields_w = torch.fft.ifft(fields, dim=-1)

        # Use consistent iteration without conditional tqdm to avoid checkpointing issues
        for _ in range(steps):
            fields, fields_w = self._one_step(fields, fields_w)

        if self.config.dispersion:
            fields = torch.fft.fft(fields_w, dim=-1)
        return fields


    # def forward(self):
    #     fields = self.fields.fields  # 시작 상태 텐서

    #     if self.config.num_chunks == 1:
    #         fields = self._run_steps_pure(fields, self.domain.Nz)
    #     else:
    #         steps_remaining = self.domain.Nz
    #         for _ in range(self.config.num_chunks):
    #             L = min(self.checkpoint_interval, steps_remaining)
    #             if L <= 0: 
    #                 break
    #             # Create a pure function that doesn't modify self state
    #             def run_chunk_pure(fields_input):
    #                 return self._run_steps_pure(fields_input, L)
                
    #             fields = checkpoint(
    #                 run_chunk_pure,
    #                 fields, use_reentrant=False
    #             )
    #             steps_remaining = steps_remaining - L

    #     # === 여기 '밖'에서만 모듈 상태/저장 갱신 ===
    #     self.fields.fields = fields
    #     # saved_fields 쓰려면 전역 스텝 카운터로 한 번만 기록하거나,
    #     # checkpoint 사용 중엔 로깅을 끄는 게 안전합니다.
    #     self.output_fields = fields
    #     return fields
    def forward(self):
        fields = self.fields.fields
        
        if self.config.num_chunks == 1:
            # Simple case: no checkpointing, save snapshots directly
            fields, snapshots = self._run_with_snapshots(fields, self.domain.Nz)
        else:
            # Checkpointed case: save snapshots at regular intervals
            fields, snapshots = self._run_checkpointed_with_snapshots(fields)
        
        # Update module state
        self.fields.fields = fields
        self.output_fields = fields
        self.saved_fields = snapshots
        return fields, snapshots
    
    def _run_with_snapshots(self, fields, steps):
        """Run simulation and save snapshots without checkpointing"""
        snapshots = []
        save_indices = make_save_indices(steps, self.config.num_save)
        
        fields_w = torch.fft.ifft(fields, dim=-1)
        
        for step in range(steps):
            fields, fields_w = self._one_step(fields, fields_w)
            
            # Save snapshot if this step is in our save indices
            if step in save_indices:
                snapshots.append(fields.clone())
        
        # Always save the final state
        if steps not in save_indices:
            snapshots.append(fields.clone())
        
        if snapshots:
            snapshots = torch.stack(snapshots, dim=0)
        else:
            # Fallback: create a single snapshot with the final state
            snapshots = fields.unsqueeze(0)
            
        return fields, snapshots
    
    def _run_checkpointed_with_snapshots(self, fields):
        """Run simulation with checkpointing and save snapshots"""
        snapshots = []
        steps_remaining = self.domain.Nz
        save_indices = make_save_indices(self.domain.Nz, self.config.num_save)
        current_step = 0
        
        for chunk_idx in range(self.config.num_chunks):
            L = min(self.checkpoint_interval, steps_remaining)
            if L <= 0: 
                break
            
            # Determine which save indices fall in this chunk
            chunk_start = current_step
            chunk_end = current_step + L
            chunk_save_indices = [idx - chunk_start for idx in save_indices 
                                if chunk_start <= idx < chunk_end]
            
            def run_chunk_with_saves(fields_input):
                chunk_snapshots = []
                fields = fields_input
                fields_w = torch.fft.ifft(fields, dim=-1)
                
                for step in range(L):
                    fields, fields_w = self._one_step(fields, fields_w)
                    
                    # Save if this step corresponds to a save index
                    if step in chunk_save_indices:
                        chunk_snapshots.append(fields.clone())
                
                return fields, chunk_snapshots
            
            fields, chunk_snapshots = checkpoint(run_chunk_with_saves, fields, use_reentrant=False)
            snapshots.extend(chunk_snapshots)
            
            steps_remaining -= L
            current_step += L
        
        # Always save the final state if not already saved
        if self.domain.Nz not in save_indices:
            snapshots.append(fields.clone())
        
        if snapshots:
            snapshots = torch.stack(snapshots, dim=0)
        else:
            # Fallback: create a single snapshot with the final state
            snapshots = fields.unsqueeze(0)
            
        return fields, snapshots
    
    # def forward(self,):
        
        
    #     if self.config.num_chunks == 1:
    #         self.run(self.fields.fields)
    #     else:
    #         steps_remaining = self.domain.Nz
    #         for i in range(self.config.num_chunks):
    #             chunk_size = min(self.checkpoint_interval, steps_remaining)
    #             if chunk_size <= 0:
    #                 break
    #             checkpoint(self.run_chunk, chunk_size, use_reentrant=False)
    #             steps_remaining =  steps_remaining - chunk_size
        
    #     self.output_fields = self.fields.fields
            
