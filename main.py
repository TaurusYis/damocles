#!/usr/bin/env python3
"""
DDR S-Parameter Analysis Tool
Main script for analyzing and comparing DDR channel S-parameters
"""

import os
import numpy as np
from ddr_s_parameter_analyzer import DDRSParameterAnalyzer
from touchstone_generator import create_example_touchstone_files, SPARAMS_DIR
from touchstone_processor import DDRTouchstoneProcessor
from plotting_utils import plot_s_matrix_multi, create_excel_with_s_matrix

def create_sample_touchstone_file(filename: str, nports: int = 24, 
                                freq_start: float = 1e6, freq_stop: float = 20e9,
                                n_points: int = 1001):
    """
    Create a sample Touchstone file for testing
    
    Args:
        filename: Output filename
        nports: Number of ports
        freq_start: Start frequency (Hz)
        freq_stop: Stop frequency (Hz)
        n_points: Number of frequency points
    """
    frequencies = np.logspace(np.log10(freq_start), np.log10(freq_stop), n_points)
    
    with open(filename, 'w') as f:
        # Write header
        f.write(f"# GHz S RI R 50\n")
        f.write("! Sample DDR channel S-parameters\n")
        f.write("! Frequency(GHz) S(1,1) S(1,2) ... S(1,{}) S(2,1) ... S({},{})\n".format(nports, nports, nports))
        
        # Write data
        for freq in frequencies:
            freq_ghz = freq / 1e9
            line = f"{freq_ghz:.6f}"
            
            # Generate realistic S-parameters
            for i in range(nports):
                for j in range(nports):
                    if i == j:
                        # Diagonal elements (reflection coefficients)
                        # Realistic return loss with frequency dependence
                        s_mag = 0.1 + 0.05 * (freq_ghz / 10)  # Worse at higher frequencies
                        s_phase = np.random.uniform(-np.pi, np.pi)
                    else:
                        # Off-diagonal elements (transmission coefficients)
                        # Realistic insertion loss and crosstalk
                        if abs(i - j) == 1:
                            # Adjacent ports (higher crosstalk)
                            s_mag = 0.01 + 0.005 * (freq_ghz / 10)
                        else:
                            # Non-adjacent ports (lower crosstalk)
                            s_mag = 0.001 + 0.0005 * (freq_ghz / 10)
                        s_phase = np.random.uniform(-np.pi, np.pi)
                    
                    s_real = s_mag * np.cos(s_phase)
                    s_imag = s_mag * np.sin(s_phase)
                    line += f" {s_real:.6e} {s_imag:.6e}"
            
            f.write(line + "\n")
    
    print(f"Created sample Touchstone file: {filename}")

def main():
    """
    Main function demonstrating the DDR S-parameter analyzer
    """
    print("DDR S-Parameter Analysis Tool")
    print("=" * 50)
    
    # Check if Touchstone files exist, if not generate them
    if not os.path.exists("CPU_Socket_v1.s24p"):
        print("Touchstone files not found. Generating them first...")
        try:
            from touchstone_generator import create_example_touchstone_files
            create_example_touchstone_files()
        except ImportError:
            print("Error: Could not import touchstone_generator.py")
            print("Please run: python touchstone_generator.py")
            return
    
    # Create analyzer instance
    analyzer = DDRSParameterAnalyzer()
    
    # Load the models
    print("\nLoading S-parameter models...")
    analyzer.read_touchstone_file("CPU_Socket_v1.s24p", "CPU_Socket_v1")
    analyzer.read_touchstone_file("CPU_Socket_v2.s24p", "CPU_Socket_v2")
    analyzer.read_touchstone_file("CPU_Socket_v3.s24p", "CPU_Socket_v3")
    analyzer.read_touchstone_file("Reference_Design.s24p", "Reference_Design")
    
    # Define DDR port mapping
    # Assuming 8 DQ signals and 2 DQS differential pairs
    dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]  # DQ0-DQ7
    dqs_ports = [9, 10, 11, 12]  # DQS0+, DQS0-, DQS1+, DQS1-
    dqs_pairs = [(9, 10), (11, 12)]  # DQS0 and DQS1 differential pairs
    
    analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Generate plots for comparison
    print("\nGenerating comparison plots...")
    
    # Insertion Loss comparison
    analyzer.plot_metric_comparison('insertion_loss', 'insertion_loss_comparison.png')
    
    # Return Loss comparison
    analyzer.plot_metric_comparison('return_loss', 'return_loss_comparison.png')
    
    # TDR comparison
    analyzer.plot_metric_comparison('tdr', 'tdr_comparison.png')
    
    # Crosstalk comparison
    analyzer.plot_metric_comparison('crosstalk', 'crosstalk_comparison.png')
    
    # Export detailed results to Excel
    print("\nExporting results to Excel...")
    analyzer.export_to_excel("ddr_analysis_results.xlsx")
    analyzer.generate_comparison_report("ddr_comparison_report.xlsx")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_statistics(analyzer)
    
    print("\nAnalysis complete!")
    print("Generated files:")
    print("  - insertion_loss_comparison.png")
    print("  - return_loss_comparison.png")
    print("  - tdr_comparison.png")
    print("  - crosstalk_comparison.png")
    print("  - ddr_analysis_results.xlsx")
    print("  - ddr_comparison_report.xlsx")
    print("  - summary_statistics.txt")

def generate_summary_statistics(analyzer):
    """
    Generate summary statistics for all models
    """
    with open("summary_statistics.txt", "w") as f:
        f.write("DDR Channel Analysis Summary Statistics\n")
        f.write("=" * 50 + "\n\n")
        
        for model_name in analyzer.models.keys():
            f.write(f"{model_name}:\n")
            f.write("-" * 30 + "\n")
            
            # Insertion Loss Analysis
            il_data = analyzer.calculate_insertion_loss(model_name)
            
            # DQ Insertion Loss
            dq_ils = [il_data[f'DQ_{i}'] for i in range(1, 9)]
            dq_il_stats = {
                'mean': np.mean([np.mean(dq) for dq in dq_ils]),
                'std': np.mean([np.std(dq) for dq in dq_ils]),
                'min': np.min([np.min(dq) for dq in dq_ils]),
                'max': np.max([np.max(dq) for dq in dq_ils])
            }
            
            f.write(f"DQ Insertion Loss:\n")
            f.write(f"  Mean: {dq_il_stats['mean']:.2f} dB\n")
            f.write(f"  Std:  {dq_il_stats['std']:.2f} dB\n")
            f.write(f"  Range: {dq_il_stats['min']:.2f} to {dq_il_stats['max']:.2f} dB\n")
            
            # DQS Insertion Loss
            dqs_ils = [il_data[f'DQS_{i}_diff'] for i in range(1, 3)]
            dqs_il_stats = {
                'mean': np.mean([np.mean(dqs) for dqs in dqs_ils]),
                'std': np.mean([np.std(dqs) for dqs in dqs_ils]),
                'min': np.min([np.min(dqs) for dqs in dqs_ils]),
                'max': np.max([np.max(dqs) for dqs in dqs_ils])
            }
            
            f.write(f"DQS Insertion Loss:\n")
            f.write(f"  Mean: {dqs_il_stats['mean']:.2f} dB\n")
            f.write(f"  Std:  {dqs_il_stats['std']:.2f} dB\n")
            f.write(f"  Range: {dqs_il_stats['min']:.2f} to {dqs_il_stats['max']:.2f} dB\n")
            
            # Return Loss Analysis
            rl_data = analyzer.calculate_return_loss(model_name)
            all_rl = np.concatenate([rl_data[port] for port in rl_data.keys()])
            rl_stats = {
                'mean': np.mean(all_rl),
                'std': np.std(all_rl),
                'min': np.min(all_rl),
                'max': np.max(all_rl)
            }
            
            f.write(f"Return Loss:\n")
            f.write(f"  Mean: {rl_stats['mean']:.2f} dB\n")
            f.write(f"  Std:  {rl_stats['std']:.2f} dB\n")
            f.write(f"  Range: {rl_stats['min']:.2f} to {rl_stats['max']:.2f} dB\n")
            
            # Crosstalk Analysis
            xtalk_data = analyzer.calculate_crosstalk(model_name)
            fext_values = [values for key, values in xtalk_data.items() if key.startswith('FEXT')]
            next_values = [values for key, values in xtalk_data.items() if key.startswith('NEXT')]
            
            if fext_values:
                worst_fext = np.max([np.max(values) for values in fext_values])
                f.write(f"Worst FEXT: {worst_fext:.2f} dB\n")
            
            if next_values:
                worst_next = np.max([np.max(values) for values in next_values])
                f.write(f"Worst NEXT: {worst_next:.2f} dB\n")
            
            f.write("\n")

def analyze_real_data():
    """
    Function to analyze real Touchstone files
    Replace the file paths with your actual data
    """
    analyzer = DDRSParameterAnalyzer()
    
    # Load your real Touchstone files
    # analyzer.read_touchstone_file("path/to/your/model1.s2p", "Model1_Name")
    # analyzer.read_touchstone_file("path/to/your/model2.s2p", "Model2_Name")
    
    # Define your DDR port mapping
    # analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Generate analysis
    # analyzer.plot_metric_comparison('insertion_loss')
    # analyzer.export_to_excel("your_analysis_results.xlsx")

if __name__ == "__main__":
    # 1. Generate example Touchstone files
    create_example_touchstone_files()

    # 2. Read and process all Touchstone files in sparams/
    processor = DDRTouchstoneProcessor()
    sparam_files = [f for f in os.listdir(SPARAMS_DIR) if f.endswith('.s24p')]
    model_names = []
    for fname in sparam_files:
        model_name = os.path.splitext(fname)[0]
        processor.read_touchstone_file(os.path.join(SPARAMS_DIR, fname), model_name)
        model_names.append(model_name)
    frequencies = processor.models[model_names[0]]['frequencies']
    s_params_dict = {name: processor.models[name]['s_params'] for name in model_names}

    # 3. Plot S-matrix and save figures (multi-model, with legends and harmonic markers)
    FIGURES_DIR = "figures"
    EXCEL_FILE = "ddr_report.xlsx"
    NPORTS = 12
    IMG_WIDTH = 1200
    IMG_HEIGHT = 800
    HARMONIC_FREQ_GHZ = 2.133  # Example: DDR4-3200 fundamental (change as needed)
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_paths = plot_s_matrix_multi(
        s_params_dict, frequencies, model_names, FIGURES_DIR,
        nports=NPORTS, img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        harmonic_freq_ghz=HARMONIC_FREQ_GHZ
    )

    # 4. Create Excel report with embedded figures
    create_excel_with_s_matrix(EXCEL_FILE, plot_paths, nports=NPORTS, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    print(f"Workflow complete. See {EXCEL_FILE} and {FIGURES_DIR}/") 