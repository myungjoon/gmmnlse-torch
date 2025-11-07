import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 

plt.rcParams['font.size'] = 18

Nt = 2**14
time_window = 30 # ps
dt = time_window / Nt
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

import os
# change the working directory to the directory of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

ind = 36
data = sio.loadmat('./data/GRIN_1030_SPMGRINA_single_gpu_ss_1.mat')
# The field data is typically stored under a key like 'fields' or similar.
# Let's print the keys to inspect:
fields_init = data['prop_output']['fields'][0,0][:,:,0]
fields_final = data['prop_output']['fields'][0,0][:,:,ind]
print(f'fields shape: {fields_final.shape}')


mydata = np.load('./data/saved_fields_spm3.npy')
myfields_final = mydata[ind,:,:]

# Figure 1
plt.figure(figsize=(6, 6))
# Use a colormap with enough distinct colors for all modes, e.g., tab10 or tab20
num_modes = fields_init.shape[1]
if num_modes <= 10:
    cmap = plt.get_cmap('tab10')
elif num_modes <= 20:
    cmap = plt.get_cmap('tab20')
else:
    # fallback: use hsv for many modes
    cmap = plt.get_cmap('hsv')

for i in range(fields_init.shape[1]):
    color = cmap(i % cmap.N)
    # plt.plot(t, np.abs(fields_init[:, i])**2, linestyle='--', alpha=0.7, color=color)
    plt.plot(t, np.abs(fields_final[:, i])**2 * 1e-3, linestyle='-', label=f"Mode {i+1}", linewidth=2, color=color)
    # plt.plot(t, np.abs(myfields_final[i])**2, label=f"Final mode {i+1}", linewidth=2, alpha=0.5, color=color)
plt.xlim([-0.5, 1.0])
plt.xlabel("Time (ps)", fontsize=20)
plt.ylabel("Intensity (kW)", fontsize=20)
plt.legend(fontsize=13, ncol=2)
plt.tight_layout()


# Figure 2 
# Plot the spectrum (wavelength vs intensity) for each mode

# --- Frequency axis setup ---
c0 = 299792458  # speed of light in m/s
lambda0_nm = 1030.0  # center wavelength in nm (adjust if needed)
dt_ps = time_window / Nt
dt_s = dt_ps * 1e-12

Nt = fields_final.shape[0]
omega0 = 2 * np.pi * c0 / (lambda0_nm * 1e-9)  # central angular frequency [rad/s]
omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # relative angular frequency [rad/s]
omega_abs = omega0 + omega_rel
wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9  # wavelength axis in nm

# For plotting, sort by wavelength
# idx_sort = np.argsort(wvls_nm)
# wvls_nm_sorted = sio.loadmat('./data/lambda.mat')['lambda_nm_sorted']

# 기본 파라미터
N = 2**14
T = 30e-12         # 30 ps
dt = T / N
c = 3e8            # m/s

# 주파수축
freq = np.fft.fftfreq(N, dt)  # [Hz], -f..+f

# 중심 파장/주파수
lambda0 = 1030e-9            # 1300 nm
f0 = c / lambda0

# 절대 주파수 -> 파장 변환
freq_abs = f0 + freq
wavelength = c / freq_abs     # [m]

# nm 단위, 작은 -> 큰 정렬
wavelength_nm = np.sort(wavelength * 1e9)


plt.figure(figsize=(6, 6))
for i in range(fields_final.shape[1]):
    color = cmap(i % cmap.N)
    # FFT of the i-th mode
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(fields_final[:, i])))**2)
    # plt.plot(spectrum, label=f"Final mode {i+1}", linewidth=2, color=color)

    # spectrum_sorted = spectrum[idx_sort]
    plt.plot(wavelength_nm, spectrum, label=f"Final mode {i+1}", linewidth=2, color=color)
    plt.xlim([700, 1600])
plt.xlabel("Wavelength (nm)", fontsize=20)
plt.ylabel("Intensity", fontsize=20)
plt.title("Output Spectrum for All Modes", fontsize=22)
# plt.legend(fontsize=16, ncol=2)
plt.tight_layout()

plt.show()
