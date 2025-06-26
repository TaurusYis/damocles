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
from plotting_utils import (plot_curves_grid_multi, create_excel_with_s_matrix, 
                           generate_harmonic_marker_texts, create_excel_with_metrics_sheet,
                           generate_metrics_grid_plots, plot_s_matrix_multi)

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

def collect_ddr_metrics_data(processor, model_names, frequencies):
    """
    Use HierarchicalDataCollector to organize comprehensive DDR metrics data
    Hierarchy: [model, signal] - stores different metrics for DDR analysis
    
    DQ0-DQ7: ports 1-8 (left), 13-20 (right)
    DQS0-, DQS0+, DQS1-, DQS1+: ports 9,10,11,12 (left), 21,22,23,24 (right)
    """
    collector = HierarchicalDataCollector(["model", "signal"])
    
    dq_left_ports = list(range(1, 9))   # DQ0-DQ7 left
    dq_right_ports = list(range(13, 21)) # DQ0-DQ7 right
    dqs_left_ports = {"DQS0-": 9, "DQS0+": 10, "DQS1-": 11, "DQS1+": 12}
    dqs_right_ports = {"DQS0-": 21, "DQS0+": 22, "DQS1-": 23, "DQS1+": 24}
    dq_names = [f"DQ{i}" for i in range(8)]
    dqs_names = ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
    signal_names = dq_names + dqs_names
    
    for model_name in model_names:
        s_params = processor.models[model_name]['s_params']
        # DQ signals (left side)
        for i, dq_port in enumerate(dq_left_ports):
            port_idx = dq_port - 1
            s_curve = s_params[:, port_idx, port_idx]
            insertion_loss = -20 * np.log10(np.abs(s_curve))
            collector.add(model_name, f"DQ{i}", metric_key="insertion_loss", metric_value=insertion_loss.tolist())
            return_loss_left = -20 * np.log10(np.abs(s_curve))
            collector.add(model_name, f"DQ{i}", metric_key="return_loss_left", metric_value=return_loss_left.tolist())
            return_loss_right = return_loss_left
            collector.add(model_name, f"DQ{i}", metric_key="return_loss_right", metric_value=return_loss_right.tolist())
            # FEXT/NEXT/PSFEXT/PSNEXT (left side, simplified as before)
            fext_values = []
            for j, other_port in enumerate(dq_left_ports):
                if j != i:
                    other_idx = other_port - 1
                    fext_curve = s_params[:, other_idx, port_idx]
                    fext_db = 20 * np.log10(np.abs(fext_curve))
                    fext_values.append(fext_db)
            if fext_values:
                worst_fext = np.maximum.reduce(fext_values)
                collector.add(model_name, f"DQ{i}", metric_key="fext", metric_value=worst_fext.tolist())
                psfext = 10 * np.log10(np.sum([10**(fext/10) for fext in fext_values], axis=0))
                collector.add(model_name, f"DQ{i}", metric_key="psfext", metric_value=psfext.tolist())
            next_values = []
            for j, other_port in enumerate(dq_left_ports):
                if j != i:
                    other_idx = other_port - 1
                    next_curve = s_params[:, port_idx, other_idx]
                    next_db = 20 * np.log10(np.abs(next_curve))
                    next_values.append(next_db)
            if next_values:
                worst_next = np.maximum.reduce(next_values)
                collector.add(model_name, f"DQ{i}", metric_key="next_left", metric_value=worst_next.tolist())
                collector.add(model_name, f"DQ{i}", metric_key="next_right", metric_value=worst_next.tolist())
                psnext = 10 * np.log10(np.sum([10**(next/10) for next in next_values], axis=0))
                collector.add(model_name, f"DQ{i}", metric_key="psnext_left", metric_value=psnext.tolist())
                collector.add(model_name, f"DQ{i}", metric_key="psnext_right", metric_value=psnext.tolist())
            collector.add(model_name, f"DQ{i}", metric_key="frequency_ghz", metric_value=(frequencies/1e9).tolist())
        # DQS signals (left side)
        for dqs_name, dqs_port in dqs_left_ports.items():
            port_idx = dqs_port - 1
            s_curve = s_params[:, port_idx, port_idx]
            insertion_loss = -20 * np.log10(np.abs(s_curve))
            collector.add(model_name, dqs_name, metric_key="insertion_loss", metric_value=insertion_loss.tolist())
            return_loss_left = -20 * np.log10(np.abs(s_curve))
            collector.add(model_name, dqs_name, metric_key="return_loss_left", metric_value=return_loss_left.tolist())
            return_loss_right = return_loss_left
            collector.add(model_name, dqs_name, metric_key="return_loss_right", metric_value=return_loss_right.tolist())
            # FEXT/NEXT/PSFEXT/PSNEXT (placeholders)
            collector.add(model_name, dqs_name, metric_key="fext", metric_value=(-40 * np.ones_like(frequencies)).tolist())
            collector.add(model_name, dqs_name, metric_key="psfext", metric_value=(-40 * np.ones_like(frequencies)).tolist())
            collector.add(model_name, dqs_name, metric_key="next_left", metric_value=(-40 * np.ones_like(frequencies)).tolist())
            collector.add(model_name, dqs_name, metric_key="next_right", metric_value=(-40 * np.ones_like(frequencies)).tolist())
            collector.add(model_name, dqs_name, metric_key="psnext_left", metric_value=(-40 * np.ones_like(frequencies)).tolist())
            collector.add(model_name, dqs_name, metric_key="psnext_right", metric_value=(-40 * np.ones_like(frequencies)).tolist())
            collector.add(model_name, dqs_name, metric_key="frequency_ghz", metric_value=(frequencies/1e9).tolist())
    return collector

def extract_data_for_plotting(collector, plot_type="s_matrix", nports=12):
    """
    Extract data from collector in format suitable for plotting
    Returns: y_grid, x_data, curve_labels, grid_titles
    """
    data = collector.get()
    
    if plot_type == "s_matrix":
        # Extract S-matrix data
        models = list(data.keys())
        nports = nports  # Assuming 12 ports
        
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
    s_matrix_collector = collect_s_matrix_data(processor, model_names, frequencies, nports=24)
    
    # 4. Collect DDR metrics data
    print("Collecting DDR metrics data...")
    metrics_collector = collect_ddr_metrics_data(processor, model_names, frequencies)
    
    # 5. Save collected data
    print("Saving collected data...")
    s_matrix_collector.save_pickle("s_matrix_data.pkl")
    metrics_collector.save_pickle("metrics_data.pkl")
    s_matrix_collector.export_to_excel("s_matrix_hierarchical.xlsx")
    metrics_collector.export_to_excel("metrics_hierarchical.xlsx")
    
    # 6. Generate plots
    FIGURES_DIR = "figures"
    EXCEL_FILE = "ddr_report.xlsx"
    NPORTS = 24
    IMG_WIDTH = 1200
    IMG_HEIGHT = 800
    HARMONIC_FREQ_GHZ = 2.133
    
    S_MATRIX_FIG_DIR = "figures/s_matrix"
    METRICS_FIG_DIR = "figures/metrics"
    os.makedirs(S_MATRIX_FIG_DIR, exist_ok=True)
    os.makedirs(METRICS_FIG_DIR, exist_ok=True)
    
    # Plot S-matrix
    print("Generating S-matrix plots...")
    result = extract_data_for_plotting(s_matrix_collector, "s_matrix", nports=NPORTS)
    if result is None:
        print("Error: Failed to extract data for plotting")
        return
    
    y_grid, x_data, curve_labels, grid_titles = result
    
    if x_data is None:
        print("Warning: No frequency data found, using default frequency range")
        x_data = np.linspace(0, 20, 1001)
    
    s_params_dict = {model: processor.models[model]['s_params'] for model in model_names}
    s_matrix_plot_paths = plot_s_matrix_multi(
        s_params_dict, frequencies, model_names, S_MATRIX_FIG_DIR,
        nports=NPORTS, img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        harmonic_freq_ghz=HARMONIC_FREQ_GHZ
    )
    
    # Generate metrics plots
    print("Generating metrics plots...")
    metrics_plot_paths = generate_metrics_grid_plots(
        metrics_collector, METRICS_FIG_DIR, 
        img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        harmonic_freq_ghz=HARMONIC_FREQ_GHZ
    )
    
    # 7. Create Excel report with both sheets
    print("Creating Excel report with metrics sheet...")
    create_excel_with_metrics_sheet(EXCEL_FILE, s_matrix_plot_paths, metrics_plot_paths,
                                   nports=NPORTS, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
    
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