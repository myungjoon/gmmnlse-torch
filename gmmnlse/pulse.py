# Description: This file contains the functions to generate the pulse profiles
import numpy as np

class Pulse:
    def __init__(self, tfwhm=0, total_energy=0, t=0, p=1, C=0, t_center=0, type='custom', values=None):
        if type == 'custom':
            self.fields = values
        elif type == 'gaussian':
            self.fields = self.gaussian(tfwhm, total_energy, t, p=p, C=C, t_center=t_center)

    def gaussian(self, tfwhm, total_energy, t, p=1, C=0, t_center=0, mode_num=2):
        Nt = len(t)
        pulse_profile = np.zeros((mode_num, Nt), dtype=np.complex128)
        t0 = tfwhm / (2 * np.sqrt(np.log(2)))
        # time_profile = np.sqrt(total_energy / (t0*np.sqrt(np.pi)) * 1000) * np.exp(-(1+1j*C)*(t-t_center)**(2*p)/(2*t0**(2*p)))
        pulse_profile[0, :] = np.sqrt(total_energy / (t0 * np.sqrt(np.pi)) * 1000) * np.exp(-(1 + 1j * C) * (t - t_center)**(2 * p) / (2 * t0**(2 * p)))
        return pulse_profile