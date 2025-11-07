# Description: This file contains the functions to generate the pulse profiles
import torch
import math

# class Pulse:
#     def __init__(self, domain, coeffs, tfwhm, total_energy=1.0, p=1, C=0, t_center=0, type='gaussian', values=None):
#         num_modes = coeffs.shape[0]
#         if type == 'custom':
#             self.fields = values
#         elif type == 'gaussian':
#             self.fields = self.gaussian(domain, coeffs, tfwhm, total_energy, p=p, C=C, t_center=t_center, num_modes=num_modes)

#     def gaussian(self, domain, coeffs, tfwhm, total_energy, p=1, C=0, t_center=0, num_modes=1):
#         # pulse_profile = torch.zeros((num_modes, domain.Nt), dtype=torch.complex128)
#         t0 = tfwhm / (2 * math.sqrt(math.log(2)))
#         # time_profile = np.sqrt(total_energy / (t0*np.sqrt(np.pi)) * 1000) * np.exp(-(1+1j*C)*(t-t_center)**(2*p)/(2*t0**(2*p)))
#         pulse_profile = math.sqrt(total_energy / (t0 * math.sqrt(math.pi)) * 1000) * torch.exp(-(1 + 1j * C) * (domain.t - t_center)**(2 * p) / (2 * t0**(2 * p)))
#         pulse_profile = pulse_profile.to(torch.complex128)

#         fields = coeffs[:, None] * pulse_profile[None, :]
#         return fields



class Pulse:
    """
    coeffs: (P,) complex tensor (complex64 또는 complex128)
    domain.t: (Nt,) real tensor (float32/float64)
    """
    def __init__(self, domain, coeffs, tfwhm, total_energy=1.0,
                 p=1, C=0.0, t_center=0.0, type='gaussian', values=None):
        if type == 'custom':
            # values는 (P, Nt) complex tensor라고 가정
            self.fields = values
        elif type == 'gaussian':
            self.fields = self.gaussian(
                domain, coeffs, tfwhm, total_energy,
                p=p, C=C, t_center=t_center
            )
        else:
            raise ValueError(f"Unsupported pulse type: {type}")

    @staticmethod
    def gaussian(domain, coeffs, tfwhm, total_energy,
                 p=1.0, C=0.0, t_center=0.0):
        """
        Returns:
            fields: (P, Nt) complex tensor
        """
        # ----- dtype, device 정렬 -----
        device = coeffs.device
        cdtype = coeffs.dtype               # complex64 or complex128
        # assert cdtype in (torch.complex64, torch.complex128), "coeffs must be complex dtype"

        rdtype = torch.float32  # Always use single precision

        t = domain.t.to(device=device, dtype=rdtype)  # (Nt,)

        # ----- 스칼라/하이퍼파라미터를 torch 텐서로 -----
        tfwhm_t       = torch.as_tensor(tfwhm,       dtype=rdtype, device=device)
        total_energy_t= torch.as_tensor(total_energy,dtype=rdtype, device=device)
        p_t           = torch.as_tensor(p,           dtype=rdtype, device=device)
        C_t           = torch.as_tensor(C,           dtype=rdtype, device=device)
        t_center_t    = torch.as_tensor(t_center,    dtype=rdtype, device=device)

        # t0 = tfwhm / (2 * sqrt(log(2)))
        two = torch.as_tensor(2.0, dtype=rdtype, device=device)
        t0 = tfwhm_t / (two * torch.sqrt(torch.log(two)))

        # amplitude = sqrt(total_energy / (t0 * sqrt(pi)) * 1000)
        amp = torch.sqrt(
            total_energy_t / (t0 * torch.sqrt(torch.tensor(torch.pi, dtype=rdtype, device=device))) * 1000.0
        )  # real (rdtype)

        # phase = -(1 + i*C) * (|t - t_center|^(2p)) / (2 * t0^(2p))
        tt   = (t - t_center_t)                             # (Nt,)
        num  = torch.pow(torch.abs(tt), two * p_t)          # real
        den  = two * torch.pow(t0, two * p_t)               # real
        factor = -(torch.as_tensor(1.0, dtype=rdtype, device=device) + 1j * C_t)  # complex
        phase  = (factor * (num / den)).to(cdtype)          # complex (Nt,)

        # pulse_profile(t) = amp * exp(phase)
        pulse_profile = (amp.to(cdtype) * torch.exp(phase))  # (Nt,) complex

        # fields(P, Nt) = coeffs(P,1) * pulse_profile(1,Nt)
        fields = coeffs.unsqueeze(-1) * pulse_profile.unsqueeze(0)  # (P, Nt) complex

        return fields