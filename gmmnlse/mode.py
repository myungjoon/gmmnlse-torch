import torch


class ModeSolver:
    def __init__(self, fiber, num_modes=1, target_mode_indices=None, dtype=torch.complex128):
        self.fiber = fiber
        self.num_modes = num_modes
        self.target_mode_indices = target_mode_indices
        self.dtype = dtype

    def solve(self, ):

        mode_fields = torch.zeros((self.num_modes, self.fiber.Nx, self.fiber.Ny), dtype=self.dtype, device=self.device)
        neffs = torch.zeros((self.num_modes), dtype=self.dtype, device=self.device)

        if self.target_mode_indices is None:
            self.target_mode_indices = list(range(self.num_modes))

        for mode_index in self.target_mode_indices:
            mode_field = self.fiber.mode_field(mode_index)
            mode_field = mode_field.to(self.device)
            mode_field = mode_field.reshape(1, -1)
            mode_field = mode_field.to(self.device)

        return mode_fields, neffs