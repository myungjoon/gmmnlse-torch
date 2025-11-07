import torch
import numpy as np
import matplotlib.pyplot as plt


# Here, we have multimode field profile. Using softmax, we one to find the optized length to maximize the output power of fundamental mode.
# We have 100 propagation length, 10 modes, and each has 2**14 time domain data. Thus, total field data is [100, 10, 2**14].

def objective_function(output_fields, target_index):
    """
    Objective function to maximize the intensity of the fundamental mode (mode 0)
    at a specific spectral index (target_index) in the spectral domain.

    Args:
        output_fields: torch.Tensor, shape (num_modes, Nt), complex
        target_index: int, index in the spectral domain to maximize

    Returns:
        loss: torch.Tensor, scalar (negative intensity at [0, target_index])
    """
    # output_fields: [length, num_modes, Nt]
    # Compute the spectrum for each mode
    output_spectrum = torch.fft.fftshift(
        torch.abs(torch.fft.fft(torch.fft.ifftshift(output_fields, dim=-1))) ** 2,
        dim=-1
    )  # shape: [num_modes, Nt]
    # Take the intensity at the target index for the fundamental mode (mode 0)
    intensity = output_spectrum[:, 0, target_index]
        # softmax the output objectives
    intensity = torch.softmax(intensity, dim=0)
    # Return negative intensity (for minimization)
    loss = -intensity
    return loss




def objective_function_simplified(intensities):
    softmax_intensities = torch.softmax(intensities, dim=0)
    loss = -softmax_intensities
    return loss



intensities = torch.tensor([1, 10, 100, 1000, 1e5, 2e5, 2.05e5, 1e4, 1e4, 1e5, 1000, 1])
tau = 0.1

taus = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

# Find tau such that (max - 2nd) * tau = 3
sorted_intensities, _ = torch.sort(intensities, descending=True)
max_val = sorted_intensities[0].item()
second_val = sorted_intensities[1].item()
delta = max_val - second_val
# Find the second max (not just sorted[1] in case of duplicates)
unique_intensities = torch.unique(intensities)



if unique_intensities.numel() > 1:
    max_val = torch.max(unique_intensities).item()
    second_val = torch.max(unique_intensities[unique_intensities < max_val]).item()
    delta = max_val - second_val
    tau = 3.0 / delta

logsumexp = (1.0 / tau) * torch.logsumexp(tau * intensities, dim=0)
print(f'tau: {tau}, logsumexp: {logsumexp}')