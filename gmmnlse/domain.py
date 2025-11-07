import torch

class Domain:
    def __init__(self, Nt, Nz, dz, dt, time_window, L,):
        self.L = L
        
        self.Nt = Nt
        self.Nz = Nz
        self.dz = dz
        self.dt = dt
        self.time_window = time_window

        self.t = self.generate_grids()
        self.omega = self.generate_freqs()
        self.omega_ps = self.omega * 1e12
        

    def generate_grids(self):
        t = torch.linspace(-0.5 * self.time_window, 0.5 * self.time_window, self.Nt)
        return t
    
    def generate_freqs(self):
        dt = self.time_window / self.Nt
        omega = 2 * torch.pi * torch.fft.fftfreq(self.Nt, dt)
        return omega