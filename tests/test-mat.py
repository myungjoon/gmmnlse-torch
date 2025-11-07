import scipy.io
import matplotlib.pyplot as plt
import numpy as np

def print_mat_keys(filename):
    """
    Reads a .mat file and prints the keys (variable names) it contains.
    """
    data = scipy.io.loadmat(filename)
    print(f"Keys in '{filename}':")
    for key in data.keys():
        # Skip MATLAB internal keys
        if not key.startswith('__'):
            print(f"  {key}")

# Example usage:
# print_mat_keys(f'./data/GRIN_1030_SPMGRINA_single_gpu_mpa.mat')
data = scipy.io.loadmat(f'./data/GRIN_1030_SPMGRINA_single_gpu_mpa.mat')
fields = data['prop_output']['fields'][0,0][:,:,41]
print(f'fields shape: {fields.shape}')

plt.figure(figsize=(12, 6))
for i in range(10):
    plt.plot(np.abs(fields[:, i])**2, label=f"mode{i+1}", linewidth=2.5)
plt.xlabel("Time Index", fontsize=18)
plt.ylabel("Intensity (a.u.)", fontsize=18)
plt.xlim([-0.5, 1.0])
plt.title("Field Intensities for Modes 1 to 10", fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()

# Plot spectrum (wavelength vs intensity) for each mode
# Assume time step and center wavelength are known or can be estimated

# --- Estimate time step and center wavelength ---
Nt = fields.shape[0]
# If you know the time window in ps, set it here. Otherwise, use a default.
time_window_ps = 30  # adjust if known
dt_ps = time_window_ps / Nt
dt_s = dt_ps * 1e-12

# Center wavelength (nm), adjust if known
lambda0_nm = 1030.0
c0 = 299792458  # m/s

# Frequency axis
omega0 = 2 * np.pi * c0 / (lambda0_nm * 1e-9)  # rad/s
omega_rel = 2 * np.pi * np.fft.fftfreq(Nt, dt_s)  # rad/s
omega_abs = omega0 + omega_rel
wvls_nm = (2 * np.pi * c0 / omega_abs) * 1e9

# For plotting, sort by wavelength
idx_sort = np.argsort(wvls_nm)
wvls_nm_sorted = wvls_nm[idx_sort]

plt.figure(figsize=(12, 6))
for i in range(10):
    # Take FFT of the i-th mode
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(fields[:, i])))**2)
    spectrum_sorted = spectrum[idx_sort]
    plt.plot(wvls_nm_sorted, spectrum_sorted, label=f"mode{i+1}", linewidth=2.5)
plt.xlabel("Wavelength (nm)", fontsize=18)
plt.ylabel("Spectral Intensity (a.u.)", fontsize=18)
plt.title("Output Spectrum for Modes 1 to 10", fontsize=20)
plt.legend(fontsize=18)
plt.tight_layout()
plt.show()
