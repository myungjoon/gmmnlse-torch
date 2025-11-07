#!/usr/bin/env python3
"""
Test script to verify device consistency across CPU and CUDA devices.
This script tests that simulations run consistently on both devices.
"""

import torch
import numpy as np
from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig

def test_device_consistency():
    """Test that simulations run consistently on CPU and CUDA."""
    
    # Test parameters
    wvl0 = 1030e-9
    L0 = 0.001  # Short length for quick test
    n2 = 2.3e-20
    fr = 0.18
    num_modes = 2  # Small number for quick test
    total_energy = 50.0  # nJ
    
    Nt = 2**10  # Small size for quick test
    time_window = 10  # ps
    dt = time_window / Nt
    tfwhm = 0.2  # ps
    
    dz = 1e-5
    z = np.arange(0, L0, dz)
    Nz = len(z)
    
    # Simple test data
    betas = torch.tensor([[0.0, 0.0, 18.9382, 0.0, 0.0, 0.0],
                         [0.0, 0.0, 18.9382, 0.0, 0.0, 0.0]], dtype=torch.float32)
    
    # Simple S tensor (identity-like)
    S = torch.zeros((2, 2, 2, 2), dtype=torch.complex64)
    for i in range(2):
        S[i, i, i, i] = 1.0
    
    # Simple Raman response
    hrw = torch.ones(Nt, dtype=torch.complex64)
    
    # Test both devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    
    results = {}
    
    for device_name in devices:
        print(f"\nTesting on device: {device_name}")
        device = torch.device(device_name)
        
        # Move data to device
        betas_device = betas.to(device)
        S_device = S.to(device)
        hrw_device = hrw.to(device)
        
        # Create coeffs on device
        coeffs = 0.01 * torch.ones(num_modes, dtype=torch.float32, device=device)
        coeffs[0] = 1.0
        coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs)**2))
        
        # Create simulation objects
        domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
        fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas_device, S=S_device, L=L0, fr=fr, hrw=hrw_device)
        initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian', values=None)
        boundary = Boundary('periodic')
        config = SimConfig(center_wavelength=wvl0, num_save=10, dispersion=True, nonlinear=True, raman=False, self_steeping=False)
        
        # Create and run simulation
        sim = Simulation(domain, fiber, initial_fields, boundary, config)
        sim.run()
        
        # Store results
        results[device_name] = {
            'output_fields': sim.output_fields.detach().cpu().numpy(),
            'saved_fields': sim.saved_fields.detach().cpu().numpy() if hasattr(sim, 'saved_fields') else None
        }
        
        print(f"  Simulation completed on {device_name}")
        print(f"  Output shape: {results[device_name]['output_fields'].shape}")
        print(f"  Output dtype: {results[device_name]['output_fields'].dtype}")
    
    # Compare results if multiple devices tested
    if len(devices) > 1:
        print("\nComparing results between devices:")
        cpu_result = results['cpu']['output_fields']
        cuda_result = results['cuda']['output_fields']
        
        # Check if results are close
        max_diff = np.max(np.abs(cpu_result - cuda_result))
        mean_diff = np.mean(np.abs(cpu_result - cuda_result))
        
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        if max_diff < 1e-6:
            print("  ✓ Results are consistent between CPU and CUDA")
        else:
            print("  ✗ Results differ between CPU and CUDA")
            return False
    else:
        print("\nOnly CPU available for testing")
    
    print("\n✓ Device consistency test passed!")
    return True

if __name__ == "__main__":
    success = test_device_consistency()
    if success:
        print("\nAll tests passed! Device consistency is working properly.")
    else:
        print("\nTests failed! There may be device consistency issues.")
