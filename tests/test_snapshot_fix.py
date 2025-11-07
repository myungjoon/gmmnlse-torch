#!/usr/bin/env python3
"""
Test script to verify the snapshot saving fix works correctly.
"""

import torch
import numpy as np
from gmmnlse import Domain, Pulse, Fiber, Boundary, Simulation, SimConfig

def test_snapshot_saving():
    """Test that snapshot saving works without the 'stack expects non-empty TensorList' error"""
    
    # Simple test parameters
    L0 = 0.01
    dz = 1e-4
    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.18
    
    num_modes = 3
    total_energy = 100.0
    Nt = 2**10
    time_window = 10
    dt = time_window / Nt
    tfwhm = 0.2
    
    # Create simple test data
    device = torch.device("cpu")  # Use CPU for testing
    betas = torch.zeros((num_modes, 4), dtype=torch.float32, device=device)
    S = torch.zeros((num_modes, num_modes, num_modes, num_modes), dtype=torch.complex64, device=device)
    hrw = torch.zeros(Nt, dtype=torch.complex64, device=device)
    
    # Set up basic parameters
    for i in range(num_modes):
        betas[i, 0] = 1.0  # beta0
        betas[i, 1] = 0.0  # beta1
        S[i, i, i, i] = 1.0  # Self-coupling
    
    Nz = int(L0 / dz)
    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw)
    
    # Test coefficients
    coeffs = torch.tensor([1.0, 0.1, 0.01], dtype=torch.complex64, device=device)
    pulse = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0)
    boundary = Boundary('periodic')
    
    print(f"Testing with Nz={Nz} steps, num_save=10")
    
    # Test 1: No checkpointing (num_chunks=1)
    print("\n=== Test 1: No checkpointing ===")
    config1 = SimConfig(center_wavelength=wvl0, num_save=10, num_chunks=1)
    sim1 = Simulation(domain, fiber, pulse, boundary, config1)
    
    try:
        output_fields1, saved_fields1 = sim1()
        print(f"✓ Success! Output shape: {output_fields1.shape}")
        print(f"✓ Saved fields shape: {saved_fields1.shape}")
        print(f"✓ Number of snapshots: {saved_fields1.shape[0]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 2: With checkpointing (num_chunks=2)
    print("\n=== Test 2: With checkpointing ===")
    config2 = SimConfig(center_wavelength=wvl0, num_save=10, num_chunks=2)
    sim2 = Simulation(domain, fiber, pulse, boundary, config2)
    
    try:
        output_fields2, saved_fields2 = sim2()
        print(f"✓ Success! Output shape: {output_fields2.shape}")
        print(f"✓ Saved fields shape: {saved_fields2.shape}")
        print(f"✓ Number of snapshots: {saved_fields2.shape[0]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    # Test 3: Edge case - very few steps
    print("\n=== Test 3: Edge case - few steps ===")
    L0_small = 0.001
    dz_small = 1e-4
    Nz_small = int(L0_small / dz_small)
    domain_small = Domain(Nt, Nz_small, dz_small, dt, time_window, L=L0_small)
    fiber_small = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0_small, fr=fr, hrw=hrw)
    
    config3 = SimConfig(center_wavelength=wvl0, num_save=10, num_chunks=1)
    sim3 = Simulation(domain_small, fiber_small, pulse, boundary, config3)
    
    try:
        output_fields3, saved_fields3 = sim3()
        print(f"✓ Success! Output shape: {output_fields3.shape}")
        print(f"✓ Saved fields shape: {saved_fields3.shape}")
        print(f"✓ Number of snapshots: {saved_fields3.shape[0]}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The snapshot saving fix works correctly.")
    return True

if __name__ == "__main__":
    test_snapshot_saving()







