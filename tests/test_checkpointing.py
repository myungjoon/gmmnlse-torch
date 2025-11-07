#!/usr/bin/env python3
"""
Test script to demonstrate the memory efficiency of checkpointed simulation.
This script compares memory usage between regular and checkpointed simulations.
"""

import numpy as np
import torch
import psutil
import os
import gc
from gmmnlse import Domain, Pulse, Fiber, Boundary, SimConfig
from gmmnlse.simulation_checkpointed import SimulationCheckpointed

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def test_memory_efficiency():
    """Test memory efficiency of checkpointed vs non-checkpointed simulation"""
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test parameters
    wvl0 = 1030e-9
    L0_values = [0.01, 0.05, 0.1, 0.2]  # Different fiber lengths
    n2 = 2.3e-20
    fr = 0.18
    num_modes = 10
    total_energy = 600.0
    Nt = 2**14
    time_window = 30
    dt = time_window / Nt
    tfwhm = 0.2
    dz = 5e-5
    
    # Load predefined data
    S = np.load('./data/predefined_data.npz')['S']
    hrw = np.load('./data/predefined_data.npz')['hrw']
    betas = np.load('./data/predefined_data.npz')['betas']
    
    betas = torch.tensor(betas, dtype=torch.float32, device=device)
    S = torch.tensor(S, dtype=torch.complex64, device=device)
    hrw = torch.tensor(hrw, dtype=torch.complex64, device=device)
    
    initial_fields_gt = np.load('./data/initial_pulse_all.npy')
    initial_fields_gt = torch.tensor(initial_fields_gt, dtype=torch.complex64, device=device)
    initial_fields_gt = torch.transpose(initial_fields_gt, 0, 1)
    
    results = []
    
    for L0 in L0_values:
        print(f"\n{'='*50}")
        print(f"Testing L0 = {L0}m")
        print(f"{'='*50}")
        
        z = np.arange(0, L0, dz)
        Nz = len(z)
        print(f"Number of simulation steps: {Nz}")
        
        domain = Domain(Nt, Nz, dz, dt, time_window, L=L0)
        fiber = Fiber(wvl0=wvl0, n2=n2, betas=betas, S=S, L=L0, fr=fr, hrw=hrw)
        boundary = Boundary('periodic')
        
        coeffs = torch.ones(num_modes, dtype=torch.float32, device=device)
        coeffs = coeffs / torch.sqrt(torch.sum(torch.abs(coeffs) ** 2))
        
        # Test non-checkpointed version
        print("\nTesting non-checkpointed simulation...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage()
        
        try:
            config_no_checkpoint = SimConfig(
                center_wavelength=wvl0, 
                dispersion=True, 
                kerr=True, 
                raman=True, 
                self_steeping=True,
                use_checkpointing=False
            )
            
            initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, 
                                 p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
            sim = SimulationCheckpointed(domain, fiber, initial_fields, boundary, config_no_checkpoint)
            sim.run()
            
            final_memory = get_memory_usage()
            memory_usage_no_checkpoint = final_memory - initial_memory
            print(f"Memory usage (no checkpoint): {memory_usage_no_checkpoint:.2f} MB")
            success_no_checkpoint = True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Out of memory error: {e}")
                memory_usage_no_checkpoint = float('inf')
                success_no_checkpoint = False
            else:
                raise e
        
        # Test checkpointed version
        print("\nTesting checkpointed simulation...")
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage()
        
        try:
            checkpoint_segments = max(10, Nz // 100)
            config_checkpoint = SimConfig(
                center_wavelength=wvl0, 
                dispersion=True, 
                kerr=True, 
                raman=True, 
                self_steeping=True,
                use_checkpointing=True,
                checkpoint_segments=checkpoint_segments
            )
            
            initial_fields = Pulse(domain, coeffs, tfwhm=tfwhm, total_energy=total_energy, 
                                 p=1, C=0, t_center=0, type='custom', values=initial_fields_gt)
            sim = SimulationCheckpointed(domain, fiber, initial_fields, boundary, config_checkpoint)
            sim.run()
            
            final_memory = get_memory_usage()
            memory_usage_checkpoint = final_memory - initial_memory
            print(f"Memory usage (checkpoint): {memory_usage_checkpoint:.2f} MB")
            print(f"Number of checkpoint segments: {checkpoint_segments}")
            success_checkpoint = True
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"Out of memory error: {e}")
                memory_usage_checkpoint = float('inf')
                success_checkpoint = False
            else:
                raise e
        
        # Calculate memory reduction
        if success_no_checkpoint and success_checkpoint:
            reduction = (memory_usage_no_checkpoint - memory_usage_checkpoint) / memory_usage_no_checkpoint * 100
            print(f"Memory reduction: {reduction:.1f}%")
        elif success_checkpoint and not success_no_checkpoint:
            print("Checkpointed version succeeded where non-checkpointed failed!")
            reduction = 100
        else:
            print("Both versions failed")
            reduction = 0
        
        results.append({
            'L0': L0,
            'Nz': Nz,
            'memory_no_checkpoint': memory_usage_no_checkpoint,
            'memory_checkpoint': memory_usage_checkpoint,
            'reduction_percent': reduction,
            'success_no_checkpoint': success_no_checkpoint,
            'success_checkpoint': success_checkpoint
        })
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'L0 (m)':<8} {'Steps':<8} {'No CP (MB)':<12} {'CP (MB)':<10} {'Reduction':<10} {'Status'}")
    print("-" * 60)
    
    for result in results:
        status = "✓" if result['success_checkpoint'] else "✗"
        if result['memory_no_checkpoint'] == float('inf'):
            no_cp_str = "OOM"
        else:
            no_cp_str = f"{result['memory_no_checkpoint']:.1f}"
        
        if result['memory_checkpoint'] == float('inf'):
            cp_str = "OOM"
        else:
            cp_str = f"{result['memory_checkpoint']:.1f}"
        
        reduction_str = f"{result['reduction_percent']:.1f}%" if result['reduction_percent'] != 100 else "∞"
        
        print(f"{result['L0']:<8} {result['Nz']:<8} {no_cp_str:<12} {cp_str:<10} {reduction_str:<10} {status}")
    
    return results

if __name__ == "__main__":
    test_memory_efficiency()
