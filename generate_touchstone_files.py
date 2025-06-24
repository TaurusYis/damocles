#!/usr/bin/env python3
"""
Touchstone File Generator for DDR Channel Analysis
Generates realistic 24-port S-parameter files for testing the DDR analyzer
"""

import numpy as np
import os

def generate_realistic_ddr_s_parameters(nports=24, freq_start=1e6, freq_stop=20e9, n_points=1001):
    """
    Generate realistic DDR channel S-parameters
    
    Args:
        nports: Number of ports (default 24 for DDR byte)
        freq_start: Start frequency in Hz
        freq_stop: Stop frequency in Hz
        n_points: Number of frequency points
        
    Returns:
        frequencies: Frequency array
        s_params: S-parameter array of shape (n_freq, nports, nports)
    """
    
    # Create frequency array (logarithmic spacing)
    frequencies = np.logspace(np.log10(freq_start), np.log10(freq_stop), n_points)
    
    # Initialize S-parameter array
    s_params = np.zeros((n_points, nports, nports), dtype=complex)
    
    # Define DDR signal groups
    dq_ports = list(range(8))  # DQ0-DQ7 (ports 0-7)
    dqs_ports = list(range(8, 12))  # DQS0+, DQS0-, DQS1+, DQS1- (ports 8-11)
    other_ports = list(range(12, nports))  # Other signals (ports 12-23)
    
    # Generate S-parameters for each frequency point
    for f_idx, freq in enumerate(frequencies):
        freq_ghz = freq / 1e9
        
        # Base loss parameters (frequency dependent)
        base_loss_db_per_ghz = 0.5  # Base loss per GHz
        base_loss = base_loss_db_per_ghz * freq_ghz
        
        # Generate diagonal elements (reflection coefficients)
        for i in range(nports):
            # Realistic return loss that gets worse with frequency
            if i in dq_ports:
                # DQ signals have moderate return loss
                rl_base = -15 - 2 * (freq_ghz / 10)  # -15 dB at 1 GHz, -17 dB at 10 GHz
            elif i in dqs_ports:
                # DQS signals have better return loss (differential)
                rl_base = -20 - 1.5 * (freq_ghz / 10)
            else:
                # Other signals
                rl_base = -12 - 2.5 * (freq_ghz / 10)
            
            # Add some variation
            rl_variation = np.random.normal(0, 2)  # ±2 dB variation
            rl_total = rl_base + rl_variation
            
            # Convert to magnitude
            s_mag = 10**(rl_total / 20)
            s_phase = np.random.uniform(-np.pi, np.pi)
            
            s_params[f_idx, i, i] = s_mag * np.exp(1j * s_phase)
        
        # Generate off-diagonal elements (transmission coefficients)
        for i in range(nports):
            for j in range(nports):
                if i != j:
                    # Calculate distance between ports
                    port_distance = abs(i - j)
                    
                    # Determine signal types
                    i_is_dq = i in dq_ports
                    j_is_dq = j in dq_ports
                    i_is_dqs = i in dqs_ports
                    j_is_dqs = j in dqs_ports
                    
                    # Base insertion loss
                    if i_is_dq and j_is_dq:
                        # DQ to DQ transmission
                        if port_distance == 1:
                            # Adjacent DQ signals (higher crosstalk)
                            il_base = -25 - 3 * (freq_ghz / 10)
                        else:
                            # Non-adjacent DQ signals
                            il_base = -35 - 4 * (freq_ghz / 10) - 2 * port_distance
                    elif i_is_dqs and j_is_dqs:
                        # DQS to DQS transmission
                        if port_distance == 1:
                            # Adjacent DQS signals
                            il_base = -30 - 2.5 * (freq_ghz / 10)
                        else:
                            # Non-adjacent DQS signals
                            il_base = -40 - 3 * (freq_ghz / 10) - 2.5 * port_distance
                    elif (i_is_dq and j_is_dqs) or (i_is_dqs and j_is_dq):
                        # DQ to DQS transmission
                        il_base = -28 - 3.5 * (freq_ghz / 10) - 1.5 * port_distance
                    else:
                        # Other signal combinations
                        il_base = -20 - 2 * (freq_ghz / 10) - port_distance
                    
                    # Add frequency-dependent loss
                    il_total = il_base - base_loss
                    
                    # Add variation
                    il_variation = np.random.normal(0, 3)  # ±3 dB variation
                    il_total += il_variation
                    
                    # Convert to magnitude
                    s_mag = 10**(il_total / 20)
                    s_phase = np.random.uniform(-np.pi, np.pi)
                    
                    s_params[f_idx, i, j] = s_mag * np.exp(1j * s_phase)
    
    return frequencies, s_params

def write_touchstone_file(filename, frequencies, s_params, format_type='RI'):
    """
    Write S-parameters to Touchstone file
    
    Args:
        filename: Output filename
        frequencies: Frequency array
        s_params: S-parameter array
        format_type: 'RI' for Real/Imaginary, 'MA' for Magnitude/Phase
    """
    
    nports = s_params.shape[1]
    
    with open(filename, 'w') as f:
        # Write header
        f.write("# GHz S RI R 50\n")
        f.write("! DDR Channel S-Parameters\n")
        f.write(f"! Number of ports: {nports}\n")
        f.write("! Frequency(GHz) S(1,1) S(1,2) ... S(1,N) S(2,1) ... S(N,N)\n")
        
        # Write data
        for f_idx, freq in enumerate(frequencies):
            freq_ghz = freq / 1e9
            line = f"{freq_ghz:.6f}"
            
            for i in range(nports):
                for j in range(nports):
                    s_complex = s_params[f_idx, i, j]
                    
                    if format_type == 'RI':
                        # Real/Imaginary format
                        s_real = np.real(s_complex)
                        s_imag = np.imag(s_complex)
                        line += f" {s_real:.6e} {s_imag:.6e}"
                    else:
                        # Magnitude/Phase format
                        s_mag = np.abs(s_complex)
                        s_phase = np.angle(s_complex, deg=True)
                        line += f" {s_mag:.6e} {s_phase:.6e}"
            
            f.write(line + "\n")

def create_ddr_touchstone_files():
    """
    Create multiple DDR Touchstone files for comparison testing
    """
    
    print("Generating DDR Touchstone files...")
    
    # File 1: CPU Socket Version 1 (baseline)
    print("Creating CPU_Socket_v1.s24p...")
    frequencies, s_params_v1 = generate_realistic_ddr_s_parameters(
        nports=24, freq_start=1e6, freq_stop=20e9, n_points=1001
    )
    write_touchstone_file("CPU_Socket_v1.s24p", frequencies, s_params_v1)
    
    # File 2: CPU Socket Version 2 (improved design)
    print("Creating CPU_Socket_v2.s24p...")
    frequencies, s_params_v2 = generate_realistic_ddr_s_parameters(
        nports=24, freq_start=1e6, freq_stop=20e9, n_points=1001
    )
    
    # Modify v2 to be slightly better (lower loss, better return loss)
    s_params_v2_improved = s_params_v2.copy()
    for f_idx in range(s_params_v2.shape[0]):
        for i in range(s_params_v2.shape[1]):
            for j in range(s_params_v2.shape[2]):
                if i == j:
                    # Improve return loss by 2-3 dB
                    improvement = np.random.uniform(2, 3)
                    s_params_v2_improved[f_idx, i, j] *= 10**(improvement / 20)
                else:
                    # Improve insertion loss by 1-2 dB
                    improvement = np.random.uniform(1, 2)
                    s_params_v2_improved[f_idx, i, j] *= 10**(improvement / 20)
    
    write_touchstone_file("CPU_Socket_v2.s24p", frequencies, s_params_v2_improved)
    
    # File 3: CPU Socket Version 3 (different design approach)
    print("Creating CPU_Socket_v3.s24p...")
    frequencies, s_params_v3 = generate_realistic_ddr_s_parameters(
        nports=24, freq_start=1e6, freq_stop=20e9, n_points=1001
    )
    
    # Modify v3 to have different characteristics (better at high freq, worse at low freq)
    s_params_v3_modified = s_params_v3.copy()
    for f_idx in range(s_params_v3.shape[0]):
        freq_ghz = frequencies[f_idx] / 1e9
        freq_factor = (freq_ghz - 1) / 19  # 0 at 1 GHz, 1 at 20 GHz
        
        for i in range(s_params_v3.shape[1]):
            for j in range(s_params_v3.shape[2]):
                if i == j:
                    # Better return loss at high frequencies
                    improvement = 3 * freq_factor
                    s_params_v3_modified[f_idx, i, j] *= 10**(improvement / 20)
                else:
                    # Different insertion loss profile
                    if freq_ghz < 5:
                        # Worse at low frequencies
                        degradation = 2 * (1 - freq_factor)
                        s_params_v3_modified[f_idx, i, j] *= 10**(-degradation / 20)
                    else:
                        # Better at high frequencies
                        improvement = 2 * freq_factor
                        s_params_v3_modified[f_idx, i, j] *= 10**(improvement / 20)
    
    write_touchstone_file("CPU_Socket_v3.s24p", frequencies, s_params_v3_modified)
    
    # File 4: Reference Design (ideal case)
    print("Creating Reference_Design.s24p...")
    frequencies, s_params_ref = generate_realistic_ddr_s_parameters(
        nports=24, freq_start=1e6, freq_stop=20e9, n_points=1001
    )
    
    # Make reference design significantly better
    s_params_ref_improved = s_params_ref.copy()
    for f_idx in range(s_params_ref.shape[0]):
        for i in range(s_params_ref.shape[1]):
            for j in range(s_params_ref.shape[2]):
                if i == j:
                    # Excellent return loss
                    improvement = np.random.uniform(5, 8)
                    s_params_ref_improved[f_idx, i, j] *= 10**(improvement / 20)
                else:
                    # Excellent insertion loss
                    improvement = np.random.uniform(3, 5)
                    s_params_ref_improved[f_idx, i, j] *= 10**(improvement / 20)
    
    write_touchstone_file("Reference_Design.s24p", frequencies, s_params_ref_improved)
    
    print("\nGenerated Touchstone files:")
    print("  - CPU_Socket_v1.s24p (baseline design)")
    print("  - CPU_Socket_v2.s24p (improved design)")
    print("  - CPU_Socket_v3.s24p (different approach)")
    print("  - Reference_Design.s24p (ideal case)")
    
    # Create a summary file with file information
    with open("touchstone_files_info.txt", "w") as f:
        f.write("DDR Touchstone Files Information\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Number of ports: 24\n")
        f.write(f"Frequency range: {frequencies[0]/1e9:.3f} - {frequencies[-1]/1e9:.3f} GHz\n")
        f.write(f"Number of frequency points: {len(frequencies)}\n")
        f.write(f"Format: Real/Imaginary (RI)\n")
        f.write(f"Reference impedance: 50 ohms\n\n")
        
        f.write("File Descriptions:\n")
        f.write("- CPU_Socket_v1.s24p: Baseline CPU socket design\n")
        f.write("- CPU_Socket_v2.s24p: Improved design with better loss characteristics\n")
        f.write("- CPU_Socket_v3.s24p: Different design approach (better at high freq)\n")
        f.write("- Reference_Design.s24p: Ideal case with excellent performance\n\n")
        
        f.write("DDR Port Mapping:\n")
        f.write("- DQ signals: Ports 1-8 (DQ0-DQ7)\n")
        f.write("- DQS signals: Ports 9-12 (DQS0+, DQS0-, DQS1+, DQS1-)\n")
        f.write("- Other signals: Ports 13-24\n")
        f.write("- DQS pairs: (9,10) and (11,12)\n")
    
    print("\nCreated touchstone_files_info.txt with detailed information")

def create_small_test_files():
    """
    Create smaller test files for quick testing
    """
    print("\nGenerating small test files...")
    
    # Create smaller files for quick testing
    frequencies, s_params = generate_realistic_ddr_s_parameters(
        nports=24, freq_start=1e6, freq_stop=10e9, n_points=201
    )
    
    write_touchstone_file("test_model1.s24p", frequencies, s_params)
    
    # Create a second test file with different characteristics
    s_params_test2 = s_params.copy()
    for f_idx in range(s_params.shape[0]):
        for i in range(s_params.shape[1]):
            for j in range(s_params.shape[2]):
                if i == j:
                    # Slightly different return loss
                    s_params_test2[f_idx, i, j] *= 10**(np.random.normal(0, 1) / 20)
                else:
                    # Slightly different insertion loss
                    s_params_test2[f_idx, i, j] *= 10**(np.random.normal(0, 1) / 20)
    
    write_touchstone_file("test_model2.s24p", frequencies, s_params_test2)
    
    print("Created test files:")
    print("  - test_model1.s24p")
    print("  - test_model2.s24p")

if __name__ == "__main__":
    # Create the main DDR Touchstone files
    create_ddr_touchstone_files()
    
    # Create smaller test files
    create_small_test_files()
    
    print("\n" + "="*60)
    print("All Touchstone files generated successfully!")
    print("\nYou can now run the DDR analysis tool with these files:")
    print("  python main.py")
    print("  python example_usage.py") 