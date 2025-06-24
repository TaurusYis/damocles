#!/usr/bin/env python3
"""
Example usage of the DDR S-Parameter Analysis Tool
This script demonstrates various ways to use the analyzer
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from ddr_s_parameter_analyzer import DDRSParameterAnalyzer

def example_basic_analysis():
    """
    Basic example showing how to analyze DDR S-parameters
    """
    print("=== Basic DDR S-Parameter Analysis ===")
    
    # Check if Touchstone files exist
    if not os.path.exists("CPU_Socket_v1.s24p"):
        print("Touchstone files not found. Please run: python generate_touchstone_files.py")
        return
    
    # Create analyzer
    analyzer = DDRSParameterAnalyzer()
    
    # Load models
    print("Loading S-parameter models...")
    analyzer.read_touchstone_file("CPU_Socket_v1.s24p", "CPU_Socket_v1")
    analyzer.read_touchstone_file("CPU_Socket_v2.s24p", "CPU_Socket_v2")
    analyzer.read_touchstone_file("CPU_Socket_v3.s24p", "CPU_Socket_v3")
    analyzer.read_touchstone_file("Reference_Design.s24p", "Reference_Design")
    
    # Define DDR port mapping
    dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]  # DQ0-DQ7
    dqs_ports = [9, 10, 11, 12]  # DQS0+, DQS0-, DQS1+, DQS1-
    dqs_pairs = [(9, 10), (11, 12)]  # DQS0 and DQS1 differential pairs
    
    analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Generate all comparison plots
    print("Generating comparison plots...")
    analyzer.plot_metric_comparison('insertion_loss', 'example_insertion_loss.png')
    analyzer.plot_metric_comparison('return_loss', 'example_return_loss.png')
    analyzer.plot_metric_comparison('tdr', 'example_tdr.png')
    analyzer.plot_metric_comparison('crosstalk', 'example_crosstalk.png')
    
    # Export to Excel
    print("Exporting to Excel...")
    analyzer.export_to_excel("example_analysis.xlsx")
    analyzer.generate_comparison_report("example_comparison.xlsx")
    
    print("Basic analysis complete!")

def example_detailed_analysis():
    """
    Detailed example showing how to access specific metrics and create custom plots
    """
    print("\n=== Detailed Analysis Example ===")
    
    # Check if Touchstone files exist
    if not os.path.exists("CPU_Socket_v1.s24p"):
        print("Touchstone files not found. Please run: python generate_touchstone_files.py")
        return
    
    analyzer = DDRSParameterAnalyzer()
    
    # Load models
    analyzer.read_touchstone_file("CPU_Socket_v1.s24p", "CPU_Socket_v1")
    analyzer.read_touchstone_file("CPU_Socket_v2.s24p", "CPU_Socket_v2")
    analyzer.read_touchstone_file("CPU_Socket_v3.s24p", "CPU_Socket_v3")
    analyzer.read_touchstone_file("Reference_Design.s24p", "Reference_Design")
    
    # Define ports
    dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]
    dqs_ports = [9, 10, 11, 12]
    dqs_pairs = [(9, 10), (11, 12)]
    analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Get detailed metrics for specific model
    model_name = "CPU_Socket_v1"
    
    print(f"\nAnalyzing {model_name}:")
    
    # Insertion Loss Analysis
    il_data = analyzer.calculate_insertion_loss(model_name)
    print(f"  - DQ Insertion Loss range: {np.min(il_data['DQ_1']):.2f} to {np.max(il_data['DQ_1']):.2f} dB")
    print(f"  - DQS Differential IL range: {np.min(il_data['DQS_1_diff']):.2f} to {np.max(il_data['DQS_1_diff']):.2f} dB")
    
    # Return Loss Analysis
    rl_data = analyzer.calculate_return_loss(model_name)
    worst_rl = min([np.min(values) for values in rl_data.values()])
    print(f"  - Worst Return Loss: {worst_rl:.2f} dB")
    
    # Crosstalk Analysis
    xtalk_data = analyzer.calculate_crosstalk(model_name)
    fext_values = [values for key, values in xtalk_data.items() if key.startswith('FEXT')]
    next_values = [values for key, values in xtalk_data.items() if key.startswith('NEXT')]
    
    if fext_values:
        worst_fext = np.max([np.max(values) for values in fext_values])
        print(f"  - Worst FEXT: {worst_fext:.2f} dB")
    
    if next_values:
        worst_next = np.max([np.max(values) for values in next_values])
        print(f"  - Worst NEXT: {worst_next:.2f} dB")
    
    # Create custom comparison plot
    create_custom_comparison_plot(analyzer)

def create_custom_comparison_plot(analyzer):
    """
    Create a custom comparison plot focusing on specific metrics
    """
    print("\nCreating custom comparison plot...")
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: DQ Insertion Loss Comparison
    plt.subplot(2, 2, 1)
    for model_name in analyzer.models.keys():
        il_data = analyzer.calculate_insertion_loss(model_name)
        # Plot average DQ insertion loss
        dq_ils = [il_data[f'DQ_{i}'] for i in range(1, 9)]
        avg_dq_il = np.mean(dq_ils, axis=0)
        plt.plot(analyzer.frequencies/1e9, avg_dq_il, 
                label=f'{model_name} (Avg)', linewidth=2)
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Insertion Loss (dB)')
    plt.title('Average DQ Insertion Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Worst Return Loss
    plt.subplot(2, 2, 2)
    for model_name in analyzer.models.keys():
        rl_data = analyzer.calculate_return_loss(model_name)
        worst_rl_per_freq = np.min([rl_data[port] for port in rl_data.keys()], axis=0)
        plt.plot(analyzer.frequencies/1e9, worst_rl_per_freq, 
                label=model_name, linewidth=2)
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Return Loss (dB)')
    plt.title('Worst Return Loss Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: DQS Differential Insertion Loss
    plt.subplot(2, 2, 3)
    for model_name in analyzer.models.keys():
        il_data = analyzer.calculate_insertion_loss(model_name)
        for dqs_pair in ['DQS_1_diff', 'DQS_2_diff']:
            plt.plot(analyzer.frequencies/1e9, il_data[dqs_pair], 
                    label=f'{model_name}_{dqs_pair}', linewidth=2, linestyle='--')
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Insertion Loss (dB)')
    plt.title('DQS Differential Insertion Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Subplot 4: Crosstalk Comparison
    plt.subplot(2, 2, 4)
    for model_name in analyzer.models.keys():
        xtalk_data = analyzer.calculate_crosstalk(model_name)
        
        # Get worst-case crosstalk per frequency
        fext_values = [values for key, values in xtalk_data.items() if key.startswith('FEXT')]
        next_values = [values for key, values in xtalk_data.items() if key.startswith('NEXT')]
        
        if fext_values:
            worst_fext_per_freq = np.max([np.max(values) for values in fext_values])
            plt.plot(analyzer.frequencies/1e9, worst_fext_per_freq, 
                    label=f'{model_name}_FEXT', linewidth=2)
        
        if next_values:
            worst_next_per_freq = np.max([np.max(values) for values in next_values])
            plt.plot(analyzer.frequencies/1e9, worst_next_per_freq, 
                    label=f'{model_name}_NEXT', linewidth=2, linestyle=':')
    
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Crosstalk (dB)')
    plt.title('Worst-Case Crosstalk Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('custom_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Custom comparison plot saved as 'custom_comparison.png'")

def example_statistical_analysis():
    """
    Example showing statistical analysis of the results
    """
    print("\n=== Statistical Analysis Example ===")
    
    # Check if Touchstone files exist
    if not os.path.exists("CPU_Socket_v1.s24p"):
        print("Touchstone files not found. Please run: python generate_touchstone_files.py")
        return
    
    analyzer = DDRSParameterAnalyzer()
    
    # Load models
    analyzer.read_touchstone_file("CPU_Socket_v1.s24p", "CPU_Socket_v1")
    analyzer.read_touchstone_file("CPU_Socket_v2.s24p", "CPU_Socket_v2")
    analyzer.read_touchstone_file("CPU_Socket_v3.s24p", "CPU_Socket_v3")
    analyzer.read_touchstone_file("Reference_Design.s24p", "Reference_Design")
    
    # Define ports
    dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]
    dqs_ports = [9, 10, 11, 12]
    dqs_pairs = [(9, 10), (11, 12)]
    analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Statistical analysis
    print("\nStatistical Summary:")
    print("-" * 50)
    
    for model_name in analyzer.models.keys():
        print(f"\n{model_name}:")
        
        # Insertion Loss Statistics
        il_data = analyzer.calculate_insertion_loss(model_name)
        dq_ils = [il_data[f'DQ_{i}'] for i in range(1, 9)]
        dqs_ils = [il_data[f'DQS_{i}_diff'] for i in range(1, 3)]
        
        dq_il_stats = {
            'mean': np.mean([np.mean(dq) for dq in dq_ils]),
            'std': np.mean([np.std(dq) for dq in dq_ils]),
            'min': np.min([np.min(dq) for dq in dq_ils]),
            'max': np.max([np.max(dq) for dq in dq_ils])
        }
        
        dqs_il_stats = {
            'mean': np.mean([np.mean(dqs) for dqs in dqs_ils]),
            'std': np.mean([np.std(dqs) for dqs in dqs_ils]),
            'min': np.min([np.min(dqs) for dqs in dqs_ils]),
            'max': np.max([np.max(dqs) for dqs in dqs_ils])
        }
        
        print(f"  DQ Insertion Loss:")
        print(f"    Mean: {dq_il_stats['mean']:.2f} dB")
        print(f"    Std:  {dq_il_stats['std']:.2f} dB")
        print(f"    Range: {dq_il_stats['min']:.2f} to {dq_il_stats['max']:.2f} dB")
        
        print(f"  DQS Insertion Loss:")
        print(f"    Mean: {dqs_il_stats['mean']:.2f} dB")
        print(f"    Std:  {dqs_il_stats['std']:.2f} dB")
        print(f"    Range: {dqs_il_stats['min']:.2f} to {dqs_il_stats['max']:.2f} dB")
        
        # Return Loss Statistics
        rl_data = analyzer.calculate_return_loss(model_name)
        all_rl = np.concatenate([rl_data[port] for port in rl_data.keys()])
        rl_stats = {
            'mean': np.mean(all_rl),
            'std': np.std(all_rl),
            'min': np.min(all_rl),
            'max': np.max(all_rl)
        }
        
        print(f"  Return Loss:")
        print(f"    Mean: {rl_stats['mean']:.2f} dB")
        print(f"    Std:  {rl_stats['std']:.2f} dB")
        print(f"    Range: {rl_stats['min']:.2f} to {rl_stats['max']:.2f} dB")

def example_model_comparison():
    """
    Example showing detailed model comparison
    """
    print("\n=== Model Comparison Example ===")
    
    # Check if Touchstone files exist
    if not os.path.exists("CPU_Socket_v1.s24p"):
        print("Touchstone files not found. Please run: python generate_touchstone_files.py")
        return
    
    analyzer = DDRSParameterAnalyzer()
    
    # Load models
    analyzer.read_touchstone_file("CPU_Socket_v1.s24p", "CPU_Socket_v1")
    analyzer.read_touchstone_file("CPU_Socket_v2.s24p", "CPU_Socket_v2")
    analyzer.read_touchstone_file("CPU_Socket_v3.s24p", "CPU_Socket_v3")
    analyzer.read_touchstone_file("Reference_Design.s24p", "Reference_Design")
    
    # Define ports
    dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]
    dqs_ports = [9, 10, 11, 12]
    dqs_pairs = [(9, 10), (11, 12)]
    analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)
    
    # Compare models at specific frequencies
    target_freqs = [1, 5, 10, 15, 20]  # GHz
    
    print("\nModel Comparison at Specific Frequencies:")
    print("-" * 60)
    
    for target_freq in target_freqs:
        print(f"\nAt {target_freq} GHz:")
        
        # Find closest frequency index
        freq_idx = np.argmin(np.abs(analyzer.frequencies/1e9 - target_freq))
        actual_freq = analyzer.frequencies[freq_idx]/1e9
        
        for model_name in analyzer.models.keys():
            il_data = analyzer.calculate_insertion_loss(model_name)
            
            # Get DQ insertion loss at this frequency
            dq_ils_at_freq = [il_data[f'DQ_{i}'][freq_idx] for i in range(1, 9)]
            avg_dq_il = np.mean(dq_ils_at_freq)
            min_dq_il = np.min(dq_ils_at_freq)
            max_dq_il = np.max(dq_ils_at_freq)
            
            print(f"  {model_name}:")
            print(f"    DQ IL: {avg_dq_il:.2f} dB (range: {min_dq_il:.2f} to {max_dq_il:.2f} dB)")

if __name__ == "__main__":
    # Run all examples
    example_basic_analysis()
    example_detailed_analysis()
    example_statistical_analysis()
    example_model_comparison()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the generated files:")
    print("  - example_insertion_loss.png")
    print("  - example_return_loss.png")
    print("  - example_tdr.png")
    print("  - example_crosstalk.png")
    print("  - custom_comparison.png")
    print("  - example_analysis.xlsx")
    print("  - example_comparison.xlsx") 