"""
tdr_analyzer.py
Module for Time Domain Reflectometry (TDR) analysis of DDR S-parameters.
"""
import numpy as np
from typing import Dict, List

def calculate_tdr_from_s_params(s_params, frequencies, time_window=10e-9):
    """
    Calculate TDR from S-parameters for a single port.
    
    Args:
        s_params: S-parameters array (nfreq, nports, nports)
        frequencies: Frequency array
        time_window: Time window for TDR calculation (seconds)
    
    Returns:
        Dictionary with TDR data for each port
    """
    dt = time_window / (2 * len(frequencies))
    time_axis = np.arange(0, time_window, dt)
    tdr_results = {}
    
    for i in range(s_params.shape[1]):
        s11 = s_params[:, i, i]
        s11_padded = np.pad(s11, (0, len(s11)), mode='constant')
        window = np.hanning(len(s11_padded))
        s11_windowed = s11_padded * window
        tdr_time = np.fft.ifft(s11_windowed)
        tdr_magnitude = np.abs(tdr_time)
        n_points = min(len(tdr_magnitude), len(time_axis))
        tdr_results[f'Port_{i+1}'] = tdr_magnitude[:n_points]
    
    tdr_results['time_axis'] = time_axis[:n_points]
    return tdr_results

def collect_tdr_data(processor, model_names):
    """
    Collect TDR data for all DDR signals using the port mapping.
    
    DDR Port Mapping:
    - DQ0-DQ7: ports 1-8 (left), 13-20 (right)
    - DQS0-, DQS0+, DQS1-, DQS1+: ports 9,10,11,12 (left), 21,22,23,24 (right)
    
    Returns:
        Dictionary with TDR data organized by signal and side
    """
    tdr_data = {}
    
    # Define port mappings
    dq_left_ports = list(range(1, 9))   # DQ0-DQ7 left
    dq_right_ports = list(range(13, 21)) # DQ0-DQ7 right
    dqs_left_ports = {"DQS0-": 9, "DQS0+": 10, "DQS1-": 11, "DQS1+": 12}
    dqs_right_ports = {"DQS0-": 21, "DQS0+": 22, "DQS1-": 23, "DQS1+": 24}
    
    # Initialize data structure for all signals
    signal_names = [f"DQ{i}" for i in range(8)] + ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
    for signal_name in signal_names:
        tdr_data[signal_name] = {'left': [], 'right': [], 'time_axis': None}
    
    for model_name in model_names:
        s_params = processor.models[model_name]['s_params']
        frequencies = processor.models[model_name]['frequencies']
        
        # Calculate TDR for all ports
        tdr_all_ports = calculate_tdr_from_s_params(s_params, frequencies)
        time_axis = tdr_all_ports['time_axis']
        
        # Store time axis (same for all models)
        for signal_name in signal_names:
            tdr_data[signal_name]['time_axis'] = time_axis
        
        # Organize by signal
        for i, dq_port in enumerate(dq_left_ports):
            signal_name = f"DQ{i}"
            port_key = f"Port_{dq_port}"
            right_port_key = f"Port_{dq_right_ports[i]}"
            
            tdr_data[signal_name]['left'].append(tdr_all_ports[port_key])
            tdr_data[signal_name]['right'].append(tdr_all_ports[right_port_key])
        
        # DQS signals
        for dqs_name in dqs_left_ports:
            left_port_key = f"Port_{dqs_left_ports[dqs_name]}"
            right_port_key = f"Port_{dqs_right_ports[dqs_name]}"
            
            tdr_data[dqs_name]['left'].append(tdr_all_ports[left_port_key])
            tdr_data[dqs_name]['right'].append(tdr_all_ports[right_port_key])
    
    return tdr_data

def prepare_tdr_data_for_plotting(tdr_data, model_names):
    """
    Prepare TDR data in the format required by plot_curves_grid_multi.
    
    Args:
        tdr_data: TDR data dictionary from collect_tdr_data
        model_names: List of model names
    
    Returns:
        y_grid, x_data, curve_labels, grid_titles for left and right sides
    """
    # Signal names in order: DQ0-DQ7, DQS0-, DQS0+, DQS1-, DQS1+
    signal_names = [f"DQ{i}" for i in range(8)] + ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
    
    # Prepare data for left side (row 0) and right side (row 1)
    y_grid = [[[] for _ in range(len(signal_names))] for _ in range(2)]  # 2 rows (left/right), ncols signals
    x_data = None
    
    for col, signal_name in enumerate(signal_names):
        if signal_name in tdr_data and tdr_data[signal_name]['time_axis'] is not None:
            time_axis = tdr_data[signal_name]['time_axis']
            if x_data is None:
                x_data = time_axis * 1e9  # Convert to ns
            
            # Left side TDR (row 0)
            left_tdr_list = tdr_data[signal_name]['left']
            for i, model_name in enumerate(model_names):
                if i < len(left_tdr_list):
                    y_grid[0][col].append(left_tdr_list[i])
            
            # Right side TDR (row 1)
            right_tdr_list = tdr_data[signal_name]['right']
            for i, model_name in enumerate(model_names):
                if i < len(right_tdr_list):
                    y_grid[1][col].append(right_tdr_list[i])
    
    curve_labels = model_names
    grid_titles = [
        [f'{signal_name}\n(Left Side)' for signal_name in signal_names],  # Row 0: Left side
        [f'{signal_name}\n(Right Side)' for signal_name in signal_names]  # Row 1: Right side
    ]
    
    return y_grid, x_data, curve_labels, grid_titles 