import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def plot_temporal_evolution(fields, extent=None, xlim=None, ylim=None):
    intensity = np.abs(fields)**2
    plt.figure(figsize=(5,5))
    plt.imshow(intensity.T, aspect='auto', extent=extent, origin='lower', cmap='jet')
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

def plot_mode_field(mode_field, extent=None, xlim=None, ylim=None):
    plt.figure(figsize=(5,5))
    plt.imshow(mode_field.T, aspect='auto', extent=extent, origin='lower', cmap='jet')
    plt.xlabel('X')
    plt.ylabel('Y')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

def plot_intensity(fields, mode_fields, radius=None, extent=None, xlim=None, ylim=None, title=None):
    num_modes = fields.shape[0]
    total_fields = np.sum(np.reshape(fields, (num_modes, 1, 1, -1)) * np.reshape(mode_fields, (num_modes, 64, 64, 1)), axis=0)
    total_intensity = np.sum(np.abs(total_fields)**2, axis=-1)
    plt.figure(figsize=(5,5.5), layout='constrained')
    plt.imshow(total_intensity, aspect='auto', origin='lower', cmap='turbo', extent=extent)
    if radius is not None:
        circle = Circle((0, 0), radius, color='white', fill=False, linewidth=2.0)
        plt.gca().add_patch(circle)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, fontsize=18)

def plot_mode_energy_evolution(saved_fields, dz=None, xlim=None, ylim=None):
    num_modes = saved_fields.shape[1]
    if dz is None:
        dz = 1e-5
    z = np.arange(0, saved_fields.shape[0]) * dz
    saved_energy = np.sum(np.abs(saved_fields)**2, axis=2)
    # normalization to sum up to 1
    
    saved_energy = saved_energy / np.sum(saved_energy, axis=1)[:,None]
    plt.figure(figsize=(6,5), layout='constrained')
    for i in range(num_modes):
        plt.plot(z, saved_energy[:,i], label=f'mode {i+1}', alpha=0.8, linewidth=2.0)
    plt.legend(fontsize=15)
    plt.xlabel('Propagation Distance (mm)', fontsize=15)
    plt.ylabel('Energy', fontsize=15)
    if xlim is not None:
        plt.xlim(xlim)
    
    if ylim is None:
        ylim = [0, 1.2 * np.max(saved_energy)]
    plt.ylim(ylim)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)