import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 


Nt = 2**14
time_window = 30 # ps
dt = time_window / Nt
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)

ind = -1
data = sio.loadmat('./data/GRIN_1030_SPMGRINA_double_cpu_ss.mat')
# The field data is typically stored under a key like 'fields' or similar.
# Let's print the keys to inspect:
fields_init = data['prop_output']['fields'][0,0][:,:,0]
fields_final = data['prop_output']['fields'][0,0][:,:,ind]
print(f'fields shape: {fields_final.shape}')


mydata = np.load('./data/saved_fields_spm3.npy')
myfields_final = mydata[ind,:,:]

# Figure 1
plt.figure(figsize=(12, 6))
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
    plt.plot(t, np.abs(fields_final[:, i])**2, linestyle='--', label=f"Final mode {i+1}", linewidth=2, color=color)
    plt.plot(t, np.abs(myfields_final[i])**2, label=f"Final mode {i+1}", linewidth=2, alpha=0.5, color=color)
plt.xlim([-0.5, 1.0])
plt.xlabel("Time Index", fontsize=20)
plt.ylabel("Intensity (a.u.)", fontsize=20)
plt.legend(fontsize=18, ncol=2)
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
idx_sort = np.argsort(wvls_nm)
wvls_nm_sorted = wvls_nm[idx_sort]

plt.figure(figsize=(12, 6))
for i in range(fields_final.shape[1]):
    color = cmap(i % cmap.N)
    # FFT of the i-th mode
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(fields_final[:, i])))**2)
    plt.plot(spectrum, label=f"Final mode {i+1}", linewidth=2, color=color)
    # spectrum_sorted = spectrum[idx_sort]
    # plt.plot(wvls_nm_sorted, spectrum_sorted, label=f"Final mode {i+1}", linewidth=2, color=color)
plt.xlabel("Wavelength (nm)", fontsize=20)
plt.ylabel("Spectral Intensity (a.u.)", fontsize=20)
plt.title("Output Spectrum for All Modes", fontsize=22)
plt.legend(fontsize=16, ncol=2)
plt.tight_layout()

plt.show()
