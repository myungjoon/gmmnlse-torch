#!/usr/bin/env python3
"""
Test script to fix memory issues with checkpointed simulation
"""

import torch
import gc
import numpy as np
from gmmnlse import Domain, Pulse, Fiber, Boundary, SimConfig
from gmmnlse.simulation_checkpointed import SimulationCheckpointed

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def test_small_simulation():
    """Test with very small parameters to isolate the issue"""
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Very small parameters for testing
    L0 = 0.01  # 1cm
    dz = 5e-5
    wvl0 = 1030e-9
    n2 = 2.3e-20
    fr = 0.18
    num_modes = 10
    total_energy = 600.0
    Nt = 2**12  # Reduced from 2**14
    time_window = 30
    dt = time_window / Nt
    tfwhm = 0.2
    
    z = np.arange(0, L0, dz)
    Nz = len(z)
    
    print(f"Simulation parameters:")
    print(f"  L0: {L0}m")
    print(f"  Nt: {Nt}")
    print(f"  Nz: {Nz}")
    
    # Load data
    S = np.load('./data/predefined_data.npz')['S']
    hrw = np.load('./data/predefined_data.npz')['hrw']
    betas = np.load('./data/predefined_data.npz')['betas']
    
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.complex64, device=device)
    hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)
    
    domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
    fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw)
    
    coeffs = torch.ones(num_modes, dtype=torch.float32, device=device)
    coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
    
    initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, p=1, C=0, t_center=0, type='gaussian')
    boundary = Boundary('periodic')
    
    # Test different checkpoint configurations
    test_configs = [
        {"use_checkpointing": False, "checkpoint_segments": 1, "name": "No checkpointing"},
        {"use_checkpointing": True, "checkpoint_segments": 1, "name": "Checkpointing with 1 segment"},
        {"use_checkpointing": True, "checkpoint_segments": 2, "name": "Checkpointing with 2 segments"},
        {"use_checkpointing": True, "checkpoint_segments": 5, "name": "Checkpointing with 5 segments"},
    ]
    
    for config_params in test_configs:
        print(f"\n{'='*50}")
        print(f"Testing: {config_params['name']}")
        print(f"{'='*50}")
        
        clear_gpu_memory()
        
        try:
            config = SimConfig(
                center_wavelength=wvl0,
                dispersion=True,
                kerr=True,
                raman=True,
                self_steeping=True,
                use_checkpointing=config_params['use_checkpointing'],
                checkpoint_segments=config_params['checkpoint_segments']
            )
            
            sim = SimulationCheckpointed(domain, fiber, initial_fields, boundary, config)
            sim.run()
            
            print(f"✓ Success: {config_params['name']}")
            print(f"  Output shape: {sim.output_fields.shape}")
            
            # Check memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
                print(f"  GPU memory allocated: {memory_allocated:.2f} GB")
                print(f"  GPU memory reserved: {memory_reserved:.2f} GB")
            
        except Exception as e:
            print(f"✗ Failed: {config_params['name']}")
            print(f"  Error: {str(e)}")
            
            # Clear memory after error
            clear_gpu_memory()

if __name__ == "__main__":
    test_small_simulation()

