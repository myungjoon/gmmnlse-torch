import numpy as np
import matplotlib.pyplot as plt

# --- 1. 시뮬레이션 파라미터 및 그리드 설정 ---
N = 1024                 # 그리드 해상도
L = 100.0                # 그리드 물리적 크기
dz = 0.2                 # 전파 스텝 크기

# 비선형 및 분산 계수
gamma = 1.0              # 비선형 계수 (Kerr effect)
beta2 = -1.0             # 군속도 분산(GVD) 계수

# 공간 및 주파수 그리드
x = np.linspace(-L/2, L/2, N)
kx = 2 * np.pi * np.fft.ifftshift(np.fft.fftfreq(N, d=L/N))

# --- 2. 연산자 정의 ---
# 선형(분산) 연산자 D. 주파수 공간에서 정의됩니다.
D = beta2 / 2.0 * kx**2
# 순방향 및 역방향 선형 전파 연산자 (반 스텝)
linear_op_half = np.exp(1j * D * dz / 2.0)
inv_linear_op_half = np.exp(-1j * D * dz / 2.0)


# --- 3. 순방향 및 역방향 전파 함수 ---

def forward_step(E, dz):
    """한 스텝 순방향으로 전파합니다."""
    # 첫 번째 반 스텝 선형 전파
    E_k = np.fft.fft(E) * linear_op_half
    E_temp = np.fft.ifft(E_k)

    # 전체 스텝 비선형 전파 (N은 E_temp의 세기에 따라 결정됨)
    nonlinear_op = np.exp(1j * gamma * np.abs(E_temp)**2 * dz)
    E_temp *= nonlinear_op

    # 두 번째 반 스텝 선형 전파
    E_k = np.fft.fft(E_temp) * linear_op_half
    E_final = np.fft.ifft(E_k)
    return E_final

def backward_step(E, dz):
    """한 스텝 역방향으로 전파합니다."""
    # 첫 번째 반 스텝 역-선형 전파 (순서가 반대)
    E_k = np.fft.fft(E) * inv_linear_op_half
    E_temp = np.fft.ifft(E_k)

    # 전체 스텝 역-비선형 전파
    inv_nonlinear_op = np.exp(-1j * gamma * np.abs(E_temp)**2 * dz)
    E_temp *= inv_nonlinear_op

    # 두 번째 반 스텝 역-선형 전파
    E_k = np.fft.fft(E_temp) * inv_linear_op_half
    E_final = np.fft.ifft(E_k)
    return E_final

# --- 4. 메인 시뮬레이션 및 결과 비교 ---

# 초기 펄스 생성 (1D 가우시안)
E_initial = np.exp(-x**2)

# 순방향 전파 실행
E_forward = forward_step(E_initial, dz)

# 역방향 전파 실행
E_restored = backward_step(E_forward, dz)

# 복원 오차 계산 (Mean Squared Error)
error = np.mean(np.abs(E_initial - E_restored)**2)
print(f"초기 펄스와 복원된 펄스 간의 평균 제곱 오차: {error:.2e}")

# --- 5. 시각화 ---
plt.style.use('dark_background')
plt.figure(figsize=(12, 8))

# 펄스 세기 플롯
plt.plot(x, np.abs(E_initial)**2, 'w-', label='Initial Pulse', linewidth=3)
plt.plot(x, np.abs(E_forward)**2, 'c-', label='Forward Propagated', linewidth=2)
plt.plot(x, np.abs(E_restored)**2, 'r--', label='Restored Pulse', linewidth=3)

# 오차 플롯 (오차가 매우 작아 거의 보이지 않음)
plt.plot(x, np.abs(E_initial - E_restored), 'y-', label='Error |Initial - Restored|', alpha=0.7)

plt.title("SSFM Forward and Backward Propagation", fontsize=16)
plt.xlabel("Position (x)", fontsize=12)
plt.ylabel("Intensity / Error", fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.2)
plt.show()