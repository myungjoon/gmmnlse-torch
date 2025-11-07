import torch

# ----- 가우시안 펄스 생성: coeffs (P,) -> fields (P, Nt) -----
def gaussian_fields(domain_t, coeffs_real, tfwhm, total_energy,
                    p=1.0, C=0.0, t_center=0.0, dtype=torch.complex64):
    device = domain_t.device
    rdtype = torch.float32 if dtype==torch.complex64 else torch.float64

    tfwhm_t = torch.as_tensor(tfwhm, dtype=rdtype, device=device)
    total_E = torch.as_tensor(total_energy, dtype=rdtype, device=device)
    p_t     = torch.as_tensor(p, dtype=rdtype, device=device)
    C_t     = torch.as_tensor(C, dtype=rdtype, device=device)
    t0 = tfwhm_t / (2.0 * torch.sqrt(torch.log(torch.tensor(2.0, dtype=rdtype, device=device))))

    amp = torch.sqrt(total_E / (t0 * torch.sqrt(torch.tensor(torch.pi, dtype=rdtype, device=device))) * 1000.0)

    tt   = (domain_t - torch.as_tensor(t_center, dtype=rdtype, device=device))
    num  = (tt.abs() ** (2.0 * p_t))
    den  = 2.0 * (t0 ** (2.0 * p_t))
    phase = -(1.0 + 1j * C_t.to(rdtype)) * (num / den)
    pulse_profile = (amp * torch.exp(phase)).to(dtype)             # (Nt,)

    coeffs_c = coeffs_real.to(rdtype).to(device).unsqueeze(-1).to(dtype)  # (P,1) complex
    fields = coeffs_c * pulse_profile.unsqueeze(0)                         # (P,Nt) complex
    return fields

# ----- FFT 포함 선형 스텝 -----
def sim_step_fft(fields, H_w):
    F = torch.fft.fft(fields, dim=-1)
    F = F * H_w.unsqueeze(0)
    out = torch.fft.ifft(F, dim=-1)
    return out

# ----- 목적함수: 단일 bin(모드 합 파워) -----
def objective_sum_modes_bin(A_out, idx):
    power_modes = A_out[..., idx].abs().square().sum(dim=-1)  # (P,)
    return power_modes.sum().real

@torch.no_grad()
def fd_dirivative(f, x, u, eps):
    return (f(x + eps*u).item() - f(x - eps*u).item()) / (2.0 * eps)

# ----- coeffs 대신 theta(실수) → softplus → L2정규화 → coeffs_real 로 테스트 -----
def check_gauss_fft_theta(device="cpu", dtype=torch.complex128,
                          P=10, Nt=2048, idx=300, eps=None,
                          tfwhm=0.4, total_energy=1.0, p=1.0, C=0.0, t_center=0.0):
    torch.manual_seed(0)
    rdtype = torch.float32 if dtype==torch.complex64 else torch.float64
    default_eps = 1e-3 if dtype==torch.complex64 else 1e-6
    if eps is None: eps = default_eps

    # 시간축
    T = 10.0
    t = torch.linspace(-T/2, T/2, Nt, device=device, dtype=rdtype)

    # 주파수 응답 H_w (예시)
    w = torch.linspace(-1.0, 1.0, Nt, device=device, dtype=rdtype)
    beta = w**2
    dz   = torch.tensor(1.0, dtype=rdtype, device=device)
    c    = torch.tensor(0.5 * 1e3, dtype=rdtype, device=device)
    H_w  = torch.exp(1j * (beta * dz * c)).to(dtype)  # (Nt,)

    # 최적화 변수: theta (실수)
    theta = torch.randn(P, dtype=rdtype, device=device, requires_grad=True)

    # theta -> coeffs_real (양수+L2정규화)
    def theta_to_coeffs(theta):
        w = theta**2                       # 양수
        # w = torch.sigmoid(theta)
        # a = w / torch.linalg.vector_norm(w)
        return w                                  # (P,), real

    # 목적함수: theta -> coeffs -> fields -> sim -> obj
    def f(coeffs_real):
        coeffs_real = theta_to_coeffs(coeffs_real)
        fields = gaussian_fields(t, coeffs_real, tfwhm, total_energy, p=p, C=C,
                                 t_center=t_center, dtype=dtype)
        out = sim_step_fft(fields, H_w)
        return objective_sum_modes_bin(out, idx=idx)

    # AD grad
    L = f(theta)
    g = torch.autograd.grad(L, theta, create_graph=False)[0]

    # 방향벡터
    v = torch.randn_like(theta)
    v = v / (v.norm() + 1e-12)

    # AD 방향미분
    ad_dir = torch.dot(v, g).item()

    # FD 방향미분
    with torch.no_grad():
        fd_dir = fd_dirivative(f, theta, v, eps)

    rel_err = abs(ad_dir - fd_dir) / (abs(fd_dir) + 1e-12)
    print(f"[theta→softplus→norm→gauss→FFT→obj] AD: {ad_dir:.6e}, FD: {fd_dir:.6e}, rel.err: {rel_err:.3e}, eps={eps}")
    ok = (rel_err < (5e-2 if dtype==torch.complex64 else 5e-3))
    print("PASS?", ok)
    return ok, ad_dir, fd_dir, rel_err

# 사용 예시
check_gauss_fft_theta(device="cpu", dtype=torch.complex128, P=10, Nt=2048, idx=300)
