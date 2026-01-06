"""
Plot the propagation distance vs. output spectrum of the GMM-NLSE simulation
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18

c0 = 299792458 # m/s
wvl0 = 1030e-9
L0 = 0.05
L0_cm = int(L0*100)
dz = 1.0e-5
Nt = 2**14
time_window = 30 # ps
dt = time_window / Nt
dt_s = dt * 1e-12  # s
tfwhm = 0.2 # ps
t = np.linspace(-0.5 * time_window, 0.5 * time_window, Nt)


freq = np.fft.fftfreq(Nt, dt_s)
f0 = c0 / wvl0
freq_abs = f0 + freq
wavelength = c0 / freq_abs     
wavelength_nm = np.sort(wavelength * 1e9)

fields = np.load(f'./fields_300_5_1e-5.npy')
# fields = np.load(f'./fields_5_1e-6.npy')
print(f'fields.shape: {fields.shape}')
fields = fields
output_spectrum = np.zeros((fields.shape[0], fields.shape[2]))
for i in range(fields.shape[0]):
    output_spectrum[i] = np.fft.fftshift(np.abs(np.fft.fft(np.fft.ifftshift(fields[i,0], axes=0)))**2)
print(output_spectrum.shape)

# 2D colormap for the output spectrum, y axis is the distance, x axis is the wavelength
plt.figure(figsize=(6, 5))

# Find indices for wavelength limits
wvl_min, wvl_max = 700, 1400
wvl_mask = (wavelength_nm >= wvl_min) & (wavelength_nm <= wvl_max)
wavelength_nm_limited = wavelength_nm[wvl_mask]
output_spectrum_limited = output_spectrum[:, wvl_mask]
# output_spectrum_limited = np.log(output_spectrum_limited)

# y axis in cm, from 0 to 2 cm
num_steps = output_spectrum.shape[0]
z_cm = np.linspace(0, L0_cm, num_steps)  # 0 to 2 cm

extent = [wavelength_nm_limited[0], wavelength_nm_limited[-1], z_cm[0], z_cm[-1]]

# find the index for 1300 nm
wvl_1300_index = np.argmin(np.abs(wavelength_nm_limited - 1300))
wvl_1150_index = np.argmin(np.abs(wavelength_nm_limited - 1150))
print(f'wvl_1300_index: {wvl_1300_index}, wvl_1150_index: {wvl_1150_index}')
intensity_1300 = output_spectrum_limited[:, wvl_1300_index]
# max intensity
max_intensity_1300 = np.max(intensity_1300)
max_intensity_1300_index = np.argmax(intensity_1300)
max_intensity_1300_distance = z_cm[max_intensity_1300_index]
print(f'max_intensity_1300_distance: {max_intensity_1300_distance} cm')

# vmax as minimum of max intensity of each distance
# vmax = np.min([np.max(output_spectrum_limited[:, wvl_1300_index]), np.max(output_spectrum_limited[:, wvl_1150_index])])

plt.imshow(output_spectrum_limited, cmap='seismic', aspect='auto', origin='lower', extent=extent, )
plt.xlabel('Wavelength (nm)', fontsize=20)
plt.ylabel('Distance (cm)', fontsize=20)
plt.title('Output Spectrum', fontsize=16)
# Because wavelength_nm is not evenly spaced, set xlim to the actual min/max of wavelength_nm_limited
plt.xlim([wavelength_nm_limited[0], wavelength_nm_limited[-1]])

# Set custom x-ticks at desired wavelengths (e.g., every 50 nm)
xtick_locs = []
xtick_labels = []
for wvl in range(wvl_min, wvl_max+1, 200):
    # Find the closest wavelength in wavelength_nm_limited
    idx = np.argmin(np.abs(wavelength_nm_limited - wvl))
    xtick_locs.append(wavelength_nm_limited[idx])
    xtick_labels.append(str(wvl))
plt.xticks(xtick_locs, xtick_labels)


# plt.axvline(x=1150, color='red', linestyle='--', linewidth=2.0)
# plt.axvline(x=1300, color='cyan', linestyle='--', linewidth=2.0)
plt.tight_layout()

plt.show()