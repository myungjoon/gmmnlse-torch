import matplotlib.pyplot as plt
import numpy as np

def plot_temporal_evolution(output, extent=None, xlim=None, ylim=None):
    output_intensity = np.abs(output)**2
    plt.figure(figsize=(5,5))
    plt.imshow(output_intensity.T, aspect='auto', extent=extent, origin='lower', cmap='jet')
    plt.xlabel('Time')
    plt.ylabel('Distance')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def plot_spectral_evolution(output_intensity, extent=[-5, 5, 0, 100], xlim=None, ylim=None):
    plt.figure(figsize=(5,5))
    output_intensity = np.fft.fftshift(output_intensity, axes=0)
    plt.imshow(output_intensity.T, aspect='auto', extent=extent, origin='lower', cmap='jet')
    plt.xlabel('Frequency')
    plt.ylabel('Distance')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def plot_output_pulse(output, t, xlim=None, ylim=None):
    plt.figure(figsize=(5,5))
    plt.plot(t, np.abs(output)**2)
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)