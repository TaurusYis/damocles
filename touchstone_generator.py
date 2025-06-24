"""
touchstone_generator.py
Module for generating example Touchstone (S-parameter) files for DDR channel analysis.
"""
import os
import numpy as np

SPARAMS_DIR = "sparams"
if not os.path.exists(SPARAMS_DIR):
    os.makedirs(SPARAMS_DIR, exist_ok=True)

def generate_realistic_ddr_s_parameters(nports=24, freq_start=1e6, freq_stop=20e9, n_points=1001):
    """
    Generate realistic DDR channel S-parameters.
    Returns (frequencies, s_params) where s_params shape is (n_points, nports, nports).
    """
    frequencies = np.logspace(np.log10(freq_start), np.log10(freq_stop), n_points)
    s_params = np.zeros((n_points, nports, nports), dtype=complex)
    dq_ports = list(range(8))
    dqs_ports = list(range(8, 12))
    for f_idx, freq in enumerate(frequencies):
        freq_ghz = freq / 1e9
        base_loss_db_per_ghz = 0.5
        base_loss = base_loss_db_per_ghz * freq_ghz
        for i in range(nports):
            if i in dq_ports:
                rl_base = -15 - 2 * (freq_ghz / 10)
            elif i in dqs_ports:
                rl_base = -20 - 1.5 * (freq_ghz / 10)
            else:
                rl_base = -12 - 2.5 * (freq_ghz / 10)
            rl_variation = np.random.normal(0, 2)
            rl_total = rl_base + rl_variation
            s_mag = 10**(rl_total / 20)
            s_phase = np.random.uniform(-np.pi, np.pi)
            s_params[f_idx, i, i] = s_mag * np.exp(1j * s_phase)
        for i in range(nports):
            for j in range(nports):
                if i != j:
                    port_distance = abs(i - j)
                    i_is_dq = i in dq_ports
                    j_is_dq = j in dq_ports
                    i_is_dqs = i in dqs_ports
                    j_is_dqs = j in dqs_ports
                    if i_is_dq and j_is_dq:
                        if port_distance == 1:
                            il_base = -25 - 3 * (freq_ghz / 10)
                        else:
                            il_base = -35 - 4 * (freq_ghz / 10) - 2 * port_distance
                    elif i_is_dqs and j_is_dqs:
                        if port_distance == 1:
                            il_base = -30 - 2.5 * (freq_ghz / 10)
                        else:
                            il_base = -40 - 3 * (freq_ghz / 10) - 2.5 * port_distance
                    elif (i_is_dq and j_is_dqs) or (i_is_dqs and j_is_dq):
                        il_base = -28 - 3.5 * (freq_ghz / 10) - 1.5 * port_distance
                    else:
                        il_base = -20 - 2 * (freq_ghz / 10) - port_distance
                    il_total = il_base - base_loss
                    il_variation = np.random.normal(0, 3)
                    il_total += il_variation
                    s_mag = 10**(il_total / 20)
                    s_phase = np.random.uniform(-np.pi, np.pi)
                    s_params[f_idx, i, j] = s_mag * np.exp(1j * s_phase)
    return frequencies, s_params

def write_touchstone_file(filename, frequencies, s_params, format_type='RI'):
    """
    Write S-parameters to a Touchstone file (.sNp).
    """
    nports = s_params.shape[1]
    with open(filename, 'w') as f:
        f.write("# GHz S RI R 50\n")
        f.write("! DDR Channel S-Parameters\n")
        f.write(f"! Number of ports: {nports}\n")
        f.write("! Frequency(GHz) S(1,1) S(1,2) ... S(1,N) S(2,1) ... S(N,N)\n")
        for f_idx, freq in enumerate(frequencies):
            freq_ghz = freq / 1e9
            line = f"{freq_ghz:.6f}"
            for i in range(nports):
                for j in range(nports):
                    s_complex = s_params[f_idx, i, j]
                    if format_type == 'RI':
                        s_real = np.real(s_complex)
                        s_imag = np.imag(s_complex)
                        line += f" {s_real:.6e} {s_imag:.6e}"
                    else:
                        s_mag = np.abs(s_complex)
                        s_phase = np.angle(s_complex, deg=True)
                        line += f" {s_mag:.6e} {s_phase:.6e}"
            f.write(line + "\n")

def create_example_touchstone_files():
    """
    Create several example DDR Touchstone files in sparams/ for demonstration and testing.
    """
    print("Generating example DDR Touchstone files...")
    # Baseline
    freqs, sparams = generate_realistic_ddr_s_parameters()
    write_touchstone_file(os.path.join(SPARAMS_DIR, "CPU_Socket_v1.s24p"), freqs, sparams)
    # Improved
    freqs, sparams2 = generate_realistic_ddr_s_parameters()
    sparams2_improved = sparams2.copy()
    for f_idx in range(sparams2.shape[0]):
        for i in range(sparams2.shape[1]):
            for j in range(sparams2.shape[2]):
                if i == j:
                    sparams2_improved[f_idx, i, j] *= 10**(np.random.uniform(2, 3) / 20)
                else:
                    sparams2_improved[f_idx, i, j] *= 10**(np.random.uniform(1, 2) / 20)
    write_touchstone_file(os.path.join(SPARAMS_DIR, "CPU_Socket_v2.s24p"), freqs, sparams2_improved)
    # Different approach
    freqs, sparams3 = generate_realistic_ddr_s_parameters()
    sparams3_mod = sparams3.copy()
    for f_idx in range(sparams3.shape[0]):
        freq_ghz = freqs[f_idx] / 1e9
        freq_factor = (freq_ghz - 1) / 19
        for i in range(sparams3.shape[1]):
            for j in range(sparams3.shape[2]):
                if i == j:
                    sparams3_mod[f_idx, i, j] *= 10**((3 * freq_factor) / 20)
                else:
                    if freq_ghz < 5:
                        sparams3_mod[f_idx, i, j] *= 10**((-2 * (1 - freq_factor)) / 20)
                    else:
                        sparams3_mod[f_idx, i, j] *= 10**((2 * freq_factor) / 20)
    write_touchstone_file(os.path.join(SPARAMS_DIR, "CPU_Socket_v3.s24p"), freqs, sparams3_mod)
    # Reference
    freqs, sparams4 = generate_realistic_ddr_s_parameters()
    sparams4_mod = sparams4.copy()
    for f_idx in range(sparams4.shape[0]):
        for i in range(sparams4.shape[1]):
            for j in range(sparams4.shape[2]):
                if i == j:
                    sparams4_mod[f_idx, i, j] *= 10**(np.random.uniform(5, 8) / 20)
                else:
                    sparams4_mod[f_idx, i, j] *= 10**(np.random.uniform(3, 5) / 20)
    write_touchstone_file(os.path.join(SPARAMS_DIR, "Reference_Design.s24p"), freqs, sparams4_mod)
    print(f"Touchstone files saved in: {SPARAMS_DIR}/") 