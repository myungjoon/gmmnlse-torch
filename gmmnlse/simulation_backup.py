import torch
import math, os
from dataclasses import dataclass

from tqdm import tqdm
is_slurm_job = 'SLURM_JOB_ID' in os.environ
C0 = 299792458  # m/s

@dataclass
class SimConfig:
    center_wavelength: float
    num_save: int = 100
    dispersion: bool = True
    nonlinear: bool = True
    raman: bool = False
    self_steeping: bool = False

class Simulation:
    def __init__(self, domain, fiber, fields, boundary, config=None):
        self.domain = domain
        self.fields = fields
        self.fiber = fiber
        self.boundary = boundary
        self.config = config
        self.num_modes = fields.fields.shape[0]

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
    

    # --- linear operator D[p, ω] 구성 (분산 등) ---
    def _build_linear_operator(self):
        P, Nt = self.num_modes, self.domain.Nt
        dev = self.fields.fields.device
        # 상대 각주파수 Ω
        omega = self.domain.omega.to(device=dev, dtype=torch.float64)  # (Nt,)

        # betas[p] = [beta0, beta1, beta2, ...]
        # 기준 모드(파이썬 인덱스 0)의 beta0, beta1 실수부
        beta0_ref = float(torch.as_tensor(self.fiber.betas[0][0]).real)
        beta1_ref = float(torch.as_tensor(self.fiber.betas[0][1]).real)

        D = torch.zeros((P, Nt), dtype=torch.complex128, device=dev)

        for p in range(P):
            betap = [float(b) for b in self.fiber.betas[p]]  # 길이 ≥ 2 가정
            # k=0,1 항: 기준 모드 값 빼기
            poly = torch.zeros(Nt, dtype=torch.float64, device=dev)
            poly = poly + (betap[0] - beta0_ref)                   # β0 diff
            poly = poly + (betap[1] - beta1_ref) * omega           # β1 diff * Ω
            # k≥2 항
            for k in range(2, len(betap)):
                poly = poly + betap[k] * (omega ** k) / math.factorial(k)
            D[p] = poly.to(torch.complex128)

        return D

    def _dA_dz_instant_kerr(self, A_t, gamma):
        """
        A_t : (P, Nt) complex  (시간영역)
        gamma : scalar (i*gamma는 여기서 곱하지 않음; 호출부에서 곱해도 됨)
        반환 : (P, Nt) complex, dA/dz
        """

        SK = self.fiber.S
        # F_instant[p,t] = Σ_{q,r,s} S[p,q,r,s] * A_q * A_r * conj(A_s)
        F_inst = torch.einsum('pqrs,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())
        dA = 1j * gamma * F_inst
        return dA
    
    # --- Raman 포함 dA/dz (즉시 + Raman 지연 응답) ---
    def _dA_dz_kerr_raman(self, A_t, gamma, fr, hrw,):
        """
        A_t   : (P,Nt) complex, time domain
        gamma : scalar (W^-1 m^-1) 계수
        fr    : Raman fraction (0~1)
        hrw   : (Nt,) Raman 응답의 주파수 응답 H_R(Ω)
        """
         # (1) instant part (1-fr)
        SK = self.fiber.S  # (P,P,P,P)
        F_inst = torch.einsum('pqrs,qt,rt,st->pt', SK, A_t, A_t, A_t.conj())
        dA_inst_t = 1j * gamma * (1.0 - fr) * F_inst

        # (2) Raman delayed part
        # T[m3,m4,t] = A[m3,t] * conj(A[m4,t])  (P,P,Nt)
        T = torch.einsum('mt,nt->mnt', A_t, A_t.conj())

        # Convolution: (h_R * T)(t) = IFFT( HR(Ω) * FFT(T) )
        # HR = hrw.to(device=dev, dtype=cdtype)                  # (Nt,)
        T_w = torch.fft.fft(T, dim=-1)                         # (P,P,Nt)
        T_conv = torch.fft.ifft(T_w * hrw.unsqueeze(0).unsqueeze(0), dim=-1)  # (P,P,Nt)

        # MATLAB과 동일 스케일: Vpl = dt * fft(hrw.*ifft(Vpl)) 대응 → conv 결과에 dt 한 번 곱함
        # dt = torch.as_tensor(self.domain.dt, dtype=rdtype, device=dev)
        T_conv = T_conv * self.domain.dt

        # Raman coupling tensor SR (없으면 S로 대체)
        SR = self.fiber.S  # (P,P,P,P)

        # Vpl[p,m2,t] = Σ_{m3,m4} SR[p,m2,m3,m4] * T_conv[m3,m4,t]   (P,m2,Nt)
        Vpl = torch.einsum('pmrs,rst->pmt', SR, T_conv)

        # Up_raman[p,t] = Σ_{m2} Vpl[p,m2,t] * A_t[m2,t]             (P,Nt)
        Up_raman = torch.einsum('pmt,mt->pt', Vpl, A_t)

        dA_raman_t = 1j * gamma * fr * Up_raman

        # (3) sum in time domain
        dA = dA_inst_t + dA_raman_t

        if self.config.self_steeping:
            omega0 = 2.0 * math.pi * C0 / float(self.fiber.wvl0)  # [rad/s]
            N_total = (1.0 - fr) * F_inst + fr * Up_raman
            dA = dA + self._shock_correction(N_total, gamma, omega0)

        return dA  # time-domain

    def run(self):
        gamma = self.fiber.n2 * 2.0 * math.pi / self.fiber.wvl0
        D = self._build_linear_operator()
        # fields = self.fields.fields.clone()  # (P, Nt), complex

        print("Simulation started (RK4IP, Raman/SS ON, ifft→exp→fft convention)")

        fields = self.fields.fields
        for i in tqdm(range(self.domain.Nz - 1), disable=is_slurm_job):
            # Half-step linear dispersion : ifft → exp(i D dz/2) → fft
            fields_w = torch.fft.ifft(fields, dim=-1) 
            fields_w = fields_w * torch.exp(1j * self.domain.dz * D)           
            fields = torch.fft.fft(fields_w, dim=-1)             

            # Nonlinearity using RK4
            if self.config.nonlinear:
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
          
            # Half-step linear dispersion 
            # tmp_t2 = torch.fft.ifft(fields, dim=-1)
            # tmp_t2 = tmp_t2 * torch.exp(1j * 0.5 * self.domain.dz * D)
            # fields = torch.fft.fft(tmp_t2, dim=-1)

        self.output_fields = fields