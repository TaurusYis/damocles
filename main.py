#!/usr/bin/env python3
"""
DDR S-Parameter Analysis Tool
Main script for analyzing and comparing DDR channel S-parameters using HierarchicalDataCollector
"""

import os
import numpy as np
from data_collector import HierarchicalDataCollector
from touchstone_generator import create_example_touchstone_files, SPARAMS_DIR
from touchstone_processor import DDRTouchstoneProcessor
from plotting_utils import plot_curves_grid_multi, create_excel_with_s_matrix, generate_harmonic_marker_texts

def collect_s_matrix_data(processor, model_names, frequencies, nports=12):
    """
    Use HierarchicalDataCollector to organize S-matrix data
    Hierarchy: [model, port_pair] - stores S-parameters as metrics
    """
    collector = HierarchicalDataCollector(["model", "port_pair"])
    
    for model_name in model_names:
        s_params = processor.models[model_name]['s_params']
        
        for row in range(nports):
            for col in range(nports):
                s_curve = s_params[:, row, col]
                s_db = 20 * np.log10(np.abs(s_curve))
                
                # Store S-parameter data as metrics for this model/port_pair
                collector.add(
                    model_name, 
                    f"({row+1},{col+1})", 
                    metric_key="s_parameter_db", 
                    metric_value=s_db.tolist()
                )
                
                # Store frequency axis
                collector.add(
                    model_name,
                    f"({row+1},{col+1})",
                    metric_key="frequency_ghz",
                    metric_value=(frequencies/1e9).tolist()
                )
    
    return collector

def collect_post_processing_data(processor, model_names, frequencies):
    """
    Use HierarchicalDataCollector to organize post-processing data
    Hierarchy: [model, signal] - stores different metrics (insertion_loss, return_loss, etc.)
    """
    collector = HierarchicalDataCollector(["model", "signal"])
    
    for model_name in model_names:
        s_params = processor.models[model_name]['s_params']
        
        # Example: Calculate insertion loss for DQ signals (assuming ports 1-8 are DQ)
        for dq_port in range(1, 9):
            # S-parameter for DQ port (simplified - you'd use your actual DDR analyzer)
            s_curve = s_params[:, dq_port-1, dq_port-1]  # Reflection coefficient
            insertion_loss = -20 * np.log10(np.abs(s_curve))
            
            collector.add(
                model_name,
                f"DQ{dq_port}",
                metric_key="insertion_loss",
                metric_value=insertion_loss.tolist()
            )
            
            # Store frequency axis
            collector.add(
                model_name,
                f"DQ{dq_port}",
                metric_key="frequency_ghz",
                metric_value=(frequencies/1e9).tolist()
            )
        
        # Example: Calculate return loss
        for port in range(1, 13):
            s_curve = s_params[:, port-1, port-1]
            return_loss = -20 * np.log10(np.abs(s_curve))
            
            collector.add(
                model_name,
                f"Port{port}",
                metric_key="return_loss",
                metric_value=return_loss.tolist()
            )
            
            # Store frequency axis
            collector.add(
                model_name,
                f"Port{port}",
                metric_key="frequency_ghz",
                metric_value=(frequencies/1e9).tolist()
            )
    
    return collector

def extract_data_for_plotting(collector, plot_type="s_matrix"):
    """
    Extract data from collector in format suitable for plotting
    Returns: y_grid, x_data, curve_labels, grid_titles
    """
    data = collector.get()
    
    if plot_type == "s_matrix":
        # Extract S-matrix data
        models = list(data.keys())
        nports = 12  # Assuming 12 ports
        
        y_grid = [[[] for _ in range(nports)] for _ in range(nports)]
        curve_labels = models
        x_data = None
        
        # Get frequency axis from any model/port_pair
        for model in models:
            for port_pair in data[model]:
                if "frequency_ghz" in data[model][port_pair]:
                    x_data = data[model][port_pair]["frequency_ghz"]
                    break
            if x_data:
                break
        
        # Extract S-parameter curves
        for model in models:
            for port_pair in data[model]:
                try:
                    # Parse (row,col) format
                    row_col = port_pair.strip('()')  # Remove parentheses
                    row, col = map(int, row_col.split(','))
                    row_idx, col_idx = row - 1, col - 1
                    
                    if "s_parameter_db" in data[model][port_pair]:
                        s_data = data[model][port_pair]["s_parameter_db"]
                        if s_data:
                            y_grid[row_idx][col_idx].append(s_data)
                except:
                    continue
        
        grid_titles = [[f'S{row+1},{col+1}' for col in range(nports)] for row in range(nports)]
        
        return y_grid, x_data, curve_labels, grid_titles
    
    elif plot_type == "metrics":
        # Extract metrics data (insertion loss, return loss, etc.)
        models = list(data.keys())
        signals = set()
        metrics = set()
        
        # Collect all signals and metrics
        for model in models:
            for signal in data[model]:
                signals.add(signal)
                for metric_key in data[model][signal]:
                    if metric_key != "frequency_ghz":
                        metrics.add(metric_key)
        
        signals = sorted(list(signals))
        metrics = sorted(list(metrics))
        
        y_grid = [[[] for _ in range(len(signals))] for _ in range(len(metrics))]
        curve_labels = models
        x_data = None
        
        # Get frequency axis
        for model in models:
            for signal in data[model]:
                if "frequency_ghz" in data[model][signal]:
                    x_data = data[model][signal]["frequency_ghz"]
                    break
            if x_data:
                break
        
        # Extract curves
        for model in models:
            for signal in data[model]:
                signal_idx = signals.index(signal)
                for metric in metrics:
                    if metric in data[model][signal]:
                        metric_idx = metrics.index(metric)
                        curve_data = data[model][signal][metric]
                        if curve_data:
                            y_grid[metric_idx][signal_idx].append(curve_data)
        
        grid_titles = [[f'{metric}\n{signal}' for signal in signals] for metric in metrics]
        
        return y_grid, x_data, curve_labels, grid_titles

def main():
    print("DDR S-Parameter Analysis Tool with HierarchicalDataCollector")
    print("=" * 60)
    
    # 1. Generate example Touchstone files
    create_example_touchstone_files()
    
    # 2. Read and process all Touchstone files
    processor = DDRTouchstoneProcessor()
    sparam_files = [f for f in os.listdir(SPARAMS_DIR) if f.endswith('.s24p')]
    model_names = []
    
    for fname in sparam_files:
        model_name = os.path.splitext(fname)[0]
        processor.read_touchstone_file(os.path.join(SPARAMS_DIR, fname), model_name)
        model_names.append(model_name)
    
    frequencies = processor.models[model_names[0]]['frequencies']
    print(f"Loaded {len(model_names)} models: {model_names}")
    
    # 3. Collect S-matrix data using HierarchicalDataCollector
    print("\nCollecting S-matrix data...")
    s_matrix_collector = collect_s_matrix_data(processor, model_names, frequencies)
    
    # 4. Collect post-processing data
    print("Collecting post-processing data...")
    metrics_collector = collect_post_processing_data(processor, model_names, frequencies)
    
    # 5. Save collected data
    print("Saving collected data...")
    s_matrix_collector.save_pickle("s_matrix_data.pkl")
    metrics_collector.save_pickle("metrics_data.pkl")
    s_matrix_collector.export_to_excel("s_matrix_hierarchical.xlsx")
    metrics_collector.export_to_excel("metrics_hierarchical.xlsx")
    
    # 6. Generate plots
    FIGURES_DIR = "figures"
    EXCEL_FILE = "ddr_report.xlsx"
    NPORTS = 12
    IMG_WIDTH = 1200
    IMG_HEIGHT = 800
    HARMONIC_FREQ_GHZ = 2.133
    
    os.makedirs(FIGURES_DIR, exist_ok=True)
    
    # Plot S-matrix
    print("Generating S-matrix plots...")
    result = extract_data_for_plotting(s_matrix_collector, "s_matrix")
    if result is None:
        print("Error: Failed to extract data for plotting")
        return
    
    y_grid, x_data, curve_labels, grid_titles = result
    
    # Ensure we have frequency data
    if x_data is None:
        print("Warning: No frequency data found, using default frequency range")
        x_data = np.linspace(0, 20, 1001)  # Default 0-20 GHz range
    
    # Add harmonic markers
    marker_vlines = [HARMONIC_FREQ_GHZ, 3*HARMONIC_FREQ_GHZ] if HARMONIC_FREQ_GHZ else None
    marker_texts = generate_harmonic_marker_texts(y_grid, x_data, curve_labels, HARMONIC_FREQ_GHZ, NPORTS, NPORTS)
    
    plot_paths = plot_curves_grid_multi(
        y_grid, x_data, curve_labels, FIGURES_DIR, grid_titles=grid_titles,
        xlabel='Freq (GHz)', ylabel='S(row,col) (dB)', 
        img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        nrows=NPORTS, ncols=NPORTS, marker_texts=marker_texts, 
        marker_vlines=marker_vlines, legend_loc='best', tight_rect=[0, 0, 0.75, 1]
    )
    
    # 7. Create Excel report
    print("Creating Excel report...")
    create_excel_with_s_matrix(EXCEL_FILE, plot_paths, nports=NPORTS, 
                              img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
    
    print(f"\nWorkflow complete!")
    print(f"Generated files:")
    print(f"  - {EXCEL_FILE} (main report)")
    print(f"  - {FIGURES_DIR}/ (plot images)")
    print(f"  - s_matrix_data.pkl (pickled S-matrix data)")
    print(f"  - metrics_data.pkl (pickled metrics data)")
    print(f"  - s_matrix_hierarchical.xlsx (S-matrix data inspection)")
    print(f"  - metrics_hierarchical.xlsx (metrics data inspection)")

if __name__ == "__main__":
    main() 