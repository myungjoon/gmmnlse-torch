import torch


class Boundary:
    def __init__(self, type='periodic'):
        self.type = type

    def apply(self, field):
        if self.type == 'periodic':
            return field
        elif self.type == 'absorbing':
            return field * torch.exp(-torch.abs(field))
        else:
            raise ValueError(f"Unknown boundary condition type: {self.type}")