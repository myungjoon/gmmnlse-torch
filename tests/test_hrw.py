import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))

filename = f'./data/predefined_data_1550.npz'

hrw_ref = np.load(filename)['hrw']



Nt = 2**13
time_window = 30
dt = time_window / Nt

ts = np.linspace(0, time_window, Nt)
t1 = 12.2e-3
t2 = 32e-3

hr = ((t1**2 + t2**2) / (t1 * t2**2)) * np.sin(ts / t1) * np.exp(-ts / t2)
hrw = np.fft.ifft(hr) * Nt

plt.figure()
plt.plot(ts, hrw, label='hrw', color='k', linestyle='--', alpha=0.5)
plt.plot(ts, hrw_ref, label='hrw_ref', color='r', alpha=0.5, linestyle='-')
plt.legend()
plt.show()