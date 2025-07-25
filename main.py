#!/usr/bin/env python3
"""
DDR S-Parameter Analysis Tool
Main script for analyzing and comparing DDR channel S-parameters using modular architecture.
Now supports easy integration with external programs: you can feed your own processed data directly.

API Functions:
- run_report_from_external_data_simplified(): Simplified API - only requires S-matrix and TDR data
- run_report_from_external_data(): Full API - requires S-matrix, metrics, and TDR data
"""

import os
import numpy as np
from data_collector import HierarchicalDataCollector
from data_organizer import DataOrganizer
from plotting_utils import (
    plot_curves_grid_multi, create_excel_with_s_matrix, 
    generate_harmonic_marker_texts, create_excel_with_metrics_sheet,
    generate_metrics_grid_plots, plot_s_matrix_multi,
    generate_tdr_grid_plots
)

# Optional: Only import these if running the internal workflow
def _internal_imports():
    from touchstone_generator import create_example_touchstone_files, SPARAMS_DIR
    from touchstone_processor import DDRTouchstoneProcessor
    from s_parameter_processor import SParameterPostProcessor
    return create_example_touchstone_files, SPARAMS_DIR, DDRTouchstoneProcessor, SParameterPostProcessor

def extract_data_for_plotting(collector, plot_type="s_matrix", nports=12):
    """
    Extract data from unified collector in format suitable for plotting
    Returns: y_grid, x_data, curve_labels, grid_titles
    """
    data = collector.get()
    
    if plot_type == "s_matrix":
        # Extract S-matrix data from unified collector
        models = list(data.keys())
        nports = nports  # Assuming 12 ports
        
        y_grid = [[[] for _ in range(nports)] for _ in range(nports)]
        curve_labels = models
        x_data = None
        
        # Get frequency axis from any model/port_pair
        for model in models:
            if "s_matrix" in data[model]:
                for port_pair in data[model]["s_matrix"]:
                    if "frequency_ghz" in data[model]["s_matrix"][port_pair]:
                        x_data = data[model]["s_matrix"][port_pair]["frequency_ghz"]
                        break
                if x_data:
                    break
        
        # Extract S-parameter curves
        for model in models:
            if "s_matrix" in data[model]:
                for port_pair in data[model]["s_matrix"]:
                    try:
                        # Parse (row,col) format
                        row_col = port_pair.strip('()')  # Remove parentheses
                        row, col = map(int, row_col.split(','))
                        row_idx, col_idx = row - 1, col - 1
                        
                        if "s_parameter_db" in data[model]["s_matrix"][port_pair]:
                            s_data = data[model]["s_matrix"][port_pair]["s_parameter_db"]
                            if s_data:
                                y_grid[row_idx][col_idx].append(s_data)
                    except:
                        continue
        
        grid_titles = [[f'S{row+1},{col+1}' for col in range(nports)] for row in range(nports)]
        
        return y_grid, x_data, curve_labels, grid_titles
    
    elif plot_type == "metrics":
        # Extract metrics data from unified collector
        models = list(data.keys())
        signals = set()
        metrics = set()
        
        # Collect all signals and metrics
        for model in models:
            if "metrics" in data[model]:
                for signal in data[model]["metrics"]:
                    signals.add(signal)
                    for metric_key in data[model]["metrics"][signal]:
                        if metric_key != "frequency_ghz":
                            metrics.add(metric_key)
        
        signals = sorted(list(signals))
        metrics = sorted(list(metrics))
        
        y_grid = [[[] for _ in range(len(signals))] for _ in range(len(metrics))]
        curve_labels = models
        x_data = None
        
        # Get frequency axis
        for model in models:
            if "metrics" in data[model]:
                for signal in data[model]["metrics"]:
                    if "frequency_ghz" in data[model]["metrics"][signal]:
                        x_data = data[model]["metrics"][signal]["frequency_ghz"]
                        break
                if x_data:
                    break
        
        # Extract curves
        for model in models:
            if "metrics" in data[model]:
                for signal in data[model]["metrics"]:
                    signal_idx = signals.index(signal)
                    for metric in metrics:
                        if metric in data[model]["metrics"][signal]:
                            metric_idx = metrics.index(metric)
                            curve_data = data[model]["metrics"][signal][metric]
                            if curve_data:
                                y_grid[metric_idx][signal_idx].append(curve_data)
        
        grid_titles = [[f'{metric}\n{signal}' for signal in signals] for metric in metrics]
        
        return y_grid, x_data, curve_labels, grid_titles

def create_s_matrix_collector(collector):
    """
    Create a temporary S-matrix collector for compatibility with existing plotting functions.
    """
    from data_collector import HierarchicalDataCollector
    from data_organizer import DataOrganizer
    
    s_matrix_data = DataOrganizer.extract_s_matrix_data(collector)
    temp_collector = HierarchicalDataCollector(["model", "port_pair"])
    
    for model_name, model_data in s_matrix_data.items():
        for port_pair, metrics in model_data.items():
            for metric_key, metric_value in metrics.items():
                temp_collector.add(model_name, port_pair, metric_key=metric_key, metric_value=metric_value)
    
    return temp_collector

def create_metrics_collector(collector):
    """
    Create a temporary metrics collector for compatibility with existing plotting functions.
    """
    from data_collector import HierarchicalDataCollector
    from data_organizer import DataOrganizer
    
    metrics_data = DataOrganizer.extract_metrics_data(collector)
    temp_collector = HierarchicalDataCollector(["model", "signal"])
    
    for model_name, model_data in metrics_data.items():
        for signal_name, metrics in model_data.items():
            for metric_key, metric_value in metrics.items():
                temp_collector.add(model_name, signal_name, metric_key=metric_key, metric_value=metric_value)
    
    return temp_collector

def create_tdr_collector(collector):
    """
    Create a temporary TDR collector for compatibility with existing plotting functions.
    """
    from data_collector import HierarchicalDataCollector
    from data_organizer import DataOrganizer
    
    tdr_data = DataOrganizer.extract_tdr_data(collector)
    temp_collector = HierarchicalDataCollector(["model", "signal"])
    
    for model_name, model_data in tdr_data.items():
        for signal_name, metrics in model_data.items():
            for metric_key, metric_value in metrics.items():
                temp_collector.add(model_name, signal_name, metric_key=metric_key, metric_value=metric_value)
    
    return temp_collector

def run_report_from_external_data(s_matrix_data, metrics_data, tdr_data, model_names, output_prefix="external_report"):
    """
    Generate plots and Excel report from externally provided data.
    Args:
        s_matrix_data: dict of S-matrix data (model_name -> ...)
        metrics_data: dict of metrics data (model_name -> ...)
        tdr_data: dict of TDR data (model_name -> ...)
        model_names: list of model names (order for plotting)
        output_prefix: prefix for output files (default: 'external_report')
    """
    print("Organizing external data into collector...")
    unified_collector = DataOrganizer.organize_all_data(s_matrix_data, metrics_data, tdr_data)
    print("✓ Data organized into unified HierarchicalDataCollector")

    # Create temporary collectors for plotting
    s_matrix_collector = create_s_matrix_collector(unified_collector)
    metrics_collector = create_metrics_collector(unified_collector)
    tdr_collector = create_tdr_collector(unified_collector)

    # Output directories and filenames
    FIGURES_DIR = "figures"
    S_MATRIX_FIG_DIR = os.path.join(FIGURES_DIR, f"{output_prefix}_s_matrix")
    METRICS_FIG_DIR = os.path.join(FIGURES_DIR, f"{output_prefix}_metrics")
    TDR_FIG_DIR = os.path.join(FIGURES_DIR, f"{output_prefix}_tdr")
    os.makedirs(S_MATRIX_FIG_DIR, exist_ok=True)
    os.makedirs(METRICS_FIG_DIR, exist_ok=True)
    os.makedirs(TDR_FIG_DIR, exist_ok=True)
    EXCEL_FILE = f"{output_prefix}_report.xlsx"
    NPORTS = 24
    IMG_WIDTH = 1500
    IMG_HEIGHT = 800
    HARMONIC_FREQ_GHZ = 2.133

    # Plot S-matrix
    print("Generating S-matrix plots...")
    # For external data, you must provide s_params_dict and frequencies if you want to use plot_s_matrix_multi
    # Otherwise, fallback to grid plotting from collector
    result = extract_data_for_plotting(s_matrix_collector, "s_matrix", nports=NPORTS)
    if result is None:
        print("Error: Failed to extract data for plotting")
        return
    y_grid, x_data, curve_labels, grid_titles = result
    if x_data is None:
        print("Warning: No frequency data found, using default frequency range")
        x_data = np.linspace(0, 20, 1001)
    s_matrix_plot_paths = plot_curves_grid_multi(
        y_grid, x_data, curve_labels, S_MATRIX_FIG_DIR, grid_titles=grid_titles,
        xlabel='Freq (GHz)', ylabel='S(row,col) (dB)', img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        nrows=NPORTS, ncols=NPORTS, legend_loc='best', tight_rect=[0, 0, 0.75, 1]
    )

    # Metrics plots
    print("Generating metrics plots...")
    metrics_plot_paths = generate_metrics_grid_plots(
        metrics_collector, METRICS_FIG_DIR, 
        img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        harmonic_freq_ghz=HARMONIC_FREQ_GHZ
    )

    # TDR plots
    print("Generating TDR plots...")
    tdr_plot_paths = generate_tdr_grid_plots(
        tdr_collector, model_names, TDR_FIG_DIR,
        img_width=IMG_WIDTH, img_height=IMG_HEIGHT
    )

    # Excel report
    print("Creating Excel report...")
    create_excel_with_metrics_sheet(EXCEL_FILE, s_matrix_plot_paths, metrics_plot_paths, tdr_plot_paths,
                                   nports=NPORTS, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)
    print(f"\nExternal data report complete!\n  - {EXCEL_FILE}\n  - {FIGURES_DIR}/ (plot images)")

def run_report_from_external_data_simplified(s_matrix_data, tdr_data, model_names, output_prefix="external_report"):
    """
    Generate plots and Excel report from externally provided data (simplified API).
    Metrics are calculated internally from S-matrix data.
    
    Args:
        s_matrix_data: dict of S-matrix data (model_name -> ...)
        tdr_data: dict of TDR data (model_name -> ...)
        model_names: list of model names (order for plotting)
        output_prefix: prefix for output files (default: 'external_report')
    """
    print("Calculating metrics from S-matrix data...")
    
    # Import the post-processor for metrics calculation
    from s_parameter_processor import SParameterPostProcessor
    
    # Calculate metrics internally from S-matrix data
    post_processor = SParameterPostProcessor()
    metrics_data = {}
    
    for model_name in model_names:
        if model_name in s_matrix_data:
            # Extract S-parameters and frequencies from the provided data
            # We need to reconstruct the S-parameters array from the organized data
            s_matrix_model_data = s_matrix_data[model_name]
            
            # Find the frequency data
            frequencies = None
            for port_pair in s_matrix_model_data:
                if "frequency_ghz" in s_matrix_model_data[port_pair]:
                    frequencies = np.array(s_matrix_model_data[port_pair]["frequency_ghz"]) * 1e9  # Convert back to Hz
                    break
            
            if frequencies is None:
                print(f"Warning: No frequency data found for model {model_name}, skipping metrics calculation")
                continue
            
            # Reconstruct S-parameters array
            nports = int(np.sqrt(len(s_matrix_model_data)))
            nfreq = len(frequencies)
            s_params = np.zeros((nfreq, nports, nports), dtype=complex)
            
            for port_pair in s_matrix_model_data:
                if "s_parameter_db" in s_matrix_model_data[port_pair]:
                    try:
                        # Parse (row,col) format
                        row_col = port_pair.strip('()')  # Remove parentheses
                        row, col = map(int, row_col.split(','))
                        row_idx, col_idx = row - 1, col - 1
                        
                        # Convert dB back to linear and reconstruct complex S-parameters
                        s_db = np.array(s_matrix_model_data[port_pair]["s_parameter_db"])
                        s_mag = 10**(s_db / 20)
                        # For simplicity, assume zero phase (or you could store phase separately)
                        s_params[:, row_idx, col_idx] = s_mag
                    except:
                        continue
            
            # Calculate metrics using the post-processor
            metrics_data[model_name] = post_processor.process_ddr_metrics(s_params, frequencies, model_name)
    
    print("✓ Metrics calculated from S-matrix data")
    
    # Call the original function with all three data types (S-matrix, calculated metrics, and TDR)
    run_report_from_external_data(s_matrix_data, metrics_data, tdr_data, model_names, output_prefix)

def run_full_workflow():
    """
    Run the full internal workflow: generate example data, process, organize, plot, and report.
    """
    print("DDR S-Parameter Analysis Tool with Modular Architecture")
    print("=" * 60)
    create_example_touchstone_files, SPARAMS_DIR, DDRTouchstoneProcessor, SParameterPostProcessor = _internal_imports()

    # 1. Generate example Touchstone files (Data Generation)
    print("1. Generating example Touchstone files...")
    create_example_touchstone_files()

    # 2. Read and process all Touchstone files
    print("2. Reading Touchstone files...")
    processor = DDRTouchstoneProcessor()
    sparam_files = [f for f in os.listdir(SPARAMS_DIR) if f.endswith('.s24p')]
    model_names = []
    for fname in sparam_files:
        model_name = os.path.splitext(fname)[0]
        processor.read_touchstone_file(os.path.join(SPARAMS_DIR, fname), model_name)
        model_names.append(model_name)
    frequencies = processor.models[model_names[0]]['frequencies']
    print(f"   Loaded {len(model_names)} models: {model_names}")

    # 3. Post-process S-parameters (Post-processing)
    print("3. Post-processing S-parameters...")
    post_processor = SParameterPostProcessor()
    s_matrix_data, metrics_data, tdr_data = post_processor.process_all_models(processor, model_names)
    print("   ✓ S-matrix data processed")
    print("   ✓ DDR metrics processed")
    print("   ✓ TDR data processed (including differential TDR for DQS signals)")

    # 4. Organize data into collectors (Data Organization)
    print("4. Organizing data into collectors...")
    unified_collector = DataOrganizer.organize_all_data(
        s_matrix_data, metrics_data, tdr_data
    )
    print("   ✓ Data organized into unified HierarchicalDataCollector")

    # 5. Save collected data
    print("5. Saving collected data...")
    unified_collector.save_pickle("unified_data.pkl")
    unified_collector.export_to_excel("unified_data_hierarchical.xlsx")

    # Create temporary collectors for compatibility with existing plotting functions
    s_matrix_collector = create_s_matrix_collector(unified_collector)
    metrics_collector = create_metrics_collector(unified_collector)
    tdr_collector = create_tdr_collector(unified_collector)

    # Save individual data files for backward compatibility
    s_matrix_collector.save_pickle("s_matrix_data.pkl")
    metrics_collector.save_pickle("metrics_data.pkl")
    tdr_collector.save_pickle("tdr_data.pkl")
    s_matrix_collector.export_to_excel("s_matrix_hierarchical.xlsx")
    metrics_collector.export_to_excel("metrics_hierarchical.xlsx")
    tdr_collector.export_to_excel("tdr_hierarchical.xlsx")

    # 6. Generate plots and reports (Visualization)
    print("6. Generating plots and reports...")
    FIGURES_DIR = "figures"
    EXCEL_FILE = "ddr_report.xlsx"
    NPORTS = 24
    IMG_WIDTH = 1500
    IMG_HEIGHT = 800
    HARMONIC_FREQ_GHZ = 2.133

    S_MATRIX_FIG_DIR = "figures/s_matrix"
    METRICS_FIG_DIR = "figures/metrics"
    TDR_FIG_DIR = "figures/tdr"
    os.makedirs(S_MATRIX_FIG_DIR, exist_ok=True)
    os.makedirs(METRICS_FIG_DIR, exist_ok=True)
    os.makedirs(TDR_FIG_DIR, exist_ok=True)

    # Plot S-matrix
    print("   Generating S-matrix plots...")
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
    print("   Generating metrics plots...")
    metrics_plot_paths = generate_metrics_grid_plots(
        metrics_collector, METRICS_FIG_DIR, 
        img_width=IMG_WIDTH, img_height=IMG_HEIGHT,
        harmonic_freq_ghz=HARMONIC_FREQ_GHZ
    )

    # Generate TDR plots using the new collector-based approach
    print("   Generating TDR plots...")
    tdr_plot_paths = generate_tdr_grid_plots(
        tdr_collector, model_names, TDR_FIG_DIR,
        img_width=IMG_WIDTH, img_height=IMG_HEIGHT
    )

    # Create Excel report with all sheets
    print("   Creating Excel report...")
    create_excel_with_metrics_sheet(EXCEL_FILE, s_matrix_plot_paths, metrics_plot_paths, tdr_plot_paths,
                                   nports=NPORTS, img_width=IMG_WIDTH, img_height=IMG_HEIGHT)

    print(f"\nWorkflow complete!")
    print(f"Generated files:")
    print(f"  - {EXCEL_FILE} (main report with S-matrix, metrics, and TDR sheets)")
    print(f"  - {FIGURES_DIR}/ (plot images)")
    print(f"  - unified_data.pkl (pickled unified data)")
    print(f"  - unified_data_hierarchical.xlsx (data inspection file)")
    print(f"  - s_matrix_data.pkl (pickled S-matrix data)")
    print(f"  - metrics_data.pkl (pickled metrics data)")
    print(f"  - tdr_data.pkl (pickled TDR data)")
    print(f"  - s_matrix_hierarchical.xlsx (data inspection file)")
    print(f"  - metrics_hierarchical.xlsx (data inspection file)")
    print(f"  - tdr_hierarchical.xlsx (data inspection file)")
    print(f"\nArchitecture benefits:")
    print(f"  ✓ Clear separation of concerns")
    print(f"  ✓ Easy to replace S-parameter post-processing methods")
    print(f"  ✓ Modular and maintainable code structure")
    print(f"  ✓ Reusable plotting and data organization components")

if __name__ == "__main__":
    # By default, run the full internal workflow. To use external data, import and call run_report_from_external_data from another script.
    run_full_workflow() 