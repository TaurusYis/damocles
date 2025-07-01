"""
plotting_utils.py
Module for plotting S-parameter curves, metrics, and embedding figures in Excel.
"""
from typing import Optional, List
import os
import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np

def plot_curves_grid_multi(
    y_grid, x, curve_labels, out_dir, grid_titles=None,
    xlabel='X', ylabel='Y', img_width=1200, img_height=800,
    nrows=None, ncols=None, markers=None, marker_texts=None,
    legend_loc='best', tight_rect=[0, 0, 0.75, 1],
    marker_vlines=None, marker_box=True
):
    """
    Plot a grid of plots, each with multiple curves.
    y_grid: 2D list of lists of y-values (shape: nrows x ncols x ncurves)
    x: shared x-axis (1D array)
    curve_labels: list of curve labels (for legend)
    out_dir: directory to save images
    grid_titles: 2D list of titles for each subplot (optional)
    xlabel, ylabel: axis labels
    img_width, img_height: image size in pixels
    markers: list of lists of marker x-values for each curve (optional)
    marker_texts: 2D list of annotation strings for each subplot (optional)
    legend_loc: legend location
    tight_rect: rect for tight_layout
    marker_vlines: list of x-values for vertical lines (optional)
    marker_box: whether to show annotation box
    Returns: 2D list of image file paths
    """
    os.makedirs(out_dir, exist_ok=True)
    nrows = nrows or len(y_grid)
    ncols = ncols or len(y_grid[0])
    dpi = plt.rcParams.get('figure.dpi', 100)
    plot_paths: List[List[Optional[str]]] = [[None for _ in range(ncols)] for _ in range(nrows)]
    for row in range(nrows):
        for col in range(ncols):
            plt.figure(figsize=(img_width/dpi, img_height/dpi))
            y_curves = y_grid[row][col]
            for i, y in enumerate(y_curves):
                plt.plot(x, y, lw=2, label=curve_labels[i])
                # Optionally add markers
                if markers and markers[row][col] and markers[row][col][i]:
                    for mx in markers[row][col][i]:
                        idx = np.argmin(np.abs(x - mx))
                        plt.plot(x[idx], y[idx], marker='o', markersize=10)
            # Optionally add vertical lines
            if marker_vlines:
                for i, vline in enumerate(marker_vlines):
                    plt.axvline(x=vline, color=['red','blue','green','orange'][i%4], linestyle='--', lw=1.5)
            plt.xlabel(xlabel, fontsize=14)
            plt.ylabel(ylabel, fontsize=14)
            if grid_titles:
                plt.title(grid_titles[row][col], fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12, loc=legend_loc)
            # Optionally add annotation box
            if marker_texts and marker_texts[row][col] and marker_box:
                plt.gcf().text(1.0, 0.5, marker_texts[row][col], transform=plt.gca().transAxes, fontsize=11,
                               verticalalignment='center', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.7))
            plt.tight_layout(rect=tight_rect)
            fig_path = os.path.join(out_dir, f"plot_{row+1}_{col+1}.png")
            plt.savefig(fig_path)
            plt.close()
            plot_paths[row][col] = fig_path
    return plot_paths

def plot_s_matrix(s_params, frequencies, out_dir, nports=12, img_width=1200, img_height=800):
    """
    Plot S(row, col) for the first nports and save as images in out_dir.
    Returns a 2D list of image file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_paths: List[List[Optional[str]]] = [[None for _ in range(nports)] for _ in range(nports)]
    dpi = plt.rcParams.get('figure.dpi', 100)
    for row in range(nports):
        for col in range(nports):
            plt.figure(figsize=(img_width/dpi, img_height/dpi))
            s_curve = s_params[:, row, col]
            plt.plot(frequencies / 1e9, 20 * np.log10(np.abs(s_curve)), lw=2)
            plt.xlabel('Freq (GHz)', fontsize=14)
            plt.ylabel(f'S{row+1},{col+1} (dB)', fontsize=14)
            plt.title(f'S{row+1},{col+1}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = os.path.join(out_dir, f"S{row+1}_{col+1}.png")
            plt.savefig(fig_path)
            plt.close()
            plot_paths[row][col] = fig_path
    return plot_paths

def plot_s_matrix_multi(s_params_dict, frequencies, model_names, out_dir, nports=12, img_width=1200, img_height=800, harmonic_freq_ghz=None):
    """
    For each S(row, col), plot all models' curves together with legends, save as images in out_dir.
    If harmonic_freq_ghz is provided, add vertical line markers at 1st and 3rd harmonics and annotate dB values in groups.
    s_params_dict: dict of {model_name: s_params}
    Returns a 2D list of image file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    dpi = plt.rcParams.get('figure.dpi', 100)
    # Prepare y_grid: nports x nports x nmodels
    y_grid = [[[] for _ in range(nports)] for _ in range(nports)]
    marker_vlines = None
    if harmonic_freq_ghz is not None:
        marker_vlines = [harmonic_freq_ghz, 3*harmonic_freq_ghz]
    
    for row in range(nports):
        for col in range(nports):
            for model_name in model_names:
                s_curve = s_params_dict[model_name][:, row, col]
                y_db = 20 * np.log10(np.abs(s_curve))
                y_grid[row][col].append(y_db)
    
    x = frequencies / 1e9
    grid_titles = [[f'S{row+1},{col+1}' for col in range(nports)] for row in range(nports)]
    
    # Generate marker texts using the existing function
    marker_texts = None
    if harmonic_freq_ghz is not None:
        marker_texts = generate_harmonic_marker_texts(y_grid, x, model_names, harmonic_freq_ghz, nports, nports)
    
    return plot_curves_grid_multi(
        y_grid, x, model_names, out_dir, grid_titles=grid_titles,
        xlabel='Freq (GHz)', ylabel='S(row,col) (dB)', img_width=img_width, img_height=img_height,
        nrows=nports, ncols=nports, marker_texts=marker_texts, marker_vlines=marker_vlines,
        legend_loc='best', tight_rect=[0, 0, 0.75, 1]
    )

def create_excel_with_s_matrix(excel_file, plot_paths, nports=12, img_width=1200, img_height=800):
    """
    Create an Excel file with a sheet containing a grid of S-matrix images.
    Uses column width = img_width/7.5 and row height = img_height*0.75 for good appearance.
    """
    workbook = xlsxwriter.Workbook(excel_file)
    worksheet = workbook.add_worksheet("s-matrix")
    col_width = (img_width / 7.5) * 1.08
    row_height = img_height * 0.75
    for col in range(nports):
        worksheet.set_column(col, col, col_width)
    for row in range(nports):
        worksheet.set_row(row, row_height)
    for row in range(nports):
        for col in range(nports):
            cell = xlsxwriter.utility.xl_rowcol_to_cell(row, col)  # type: ignore
            worksheet.insert_image(cell, plot_paths[row][col],
                                  {'x_scale': 1, 'y_scale': 1, 'object_position': 1})
    worksheet2 = workbook.add_worksheet("metrics of interest")
    worksheet2.write(0, 0, "(Plots for metrics of interest will be added here)")
    workbook.close()

def generate_harmonic_marker_texts(y_grid, x_data, curve_labels, harmonic_freq_ghz, nrows, ncols):
    """
    Generate marker texts for harmonic frequencies.
    
    Args:
        y_grid: 2D list of lists of y-values
        x_data: frequency axis data
        curve_labels: list of curve labels (model names)
        harmonic_freq_ghz: fundamental harmonic frequency in GHz
        nrows, ncols: grid dimensions
    
    Returns:
        2D list of marker text strings
    """
    marker_texts = [['' for _ in range(ncols)] for _ in range(nrows)]
    
    if harmonic_freq_ghz is None or x_data is None:
        return marker_texts
    
    harmonics = [harmonic_freq_ghz, 3 * harmonic_freq_ghz]
    
    for row in range(nrows):
        for col in range(ncols):
            if y_grid[row][col]:
                marker_texts_1st = [f"1st Harmonic ({harmonics[0]:.2f} GHz):"]
                marker_texts_3rd = [f"3rd Harmonic ({harmonics[1]:.2f} GHz):"]
                
                for i, model in enumerate(curve_labels):
                    if i < len(y_grid[row][col]):
                        y_db = y_grid[row][col][i]
                        idx1 = np.argmin(np.abs(np.array(x_data) - harmonics[0]))
                        idx3 = np.argmin(np.abs(np.array(x_data) - harmonics[1]))
                        marker_texts_1st.append(f"  {model}: {y_db[idx1]:.2f} dB")
                        marker_texts_3rd.append(f"  {model}: {y_db[idx3]:.2f} dB")
                
                marker_texts[row][col] = '\n'.join(marker_texts_1st) + '\n\n' + '\n'.join(marker_texts_3rd)
    
    return marker_texts

def generate_metrics_grid_plots(collector, out_dir, img_width=1200, img_height=800, harmonic_freq_ghz=None):
    """
    Generate a grid of metrics plots using plot_curves_grid_multi.
    
    Args:
        collector: HierarchicalDataCollector with metrics data
        out_dir: Directory to save plots
        img_width, img_height: Image dimensions
        harmonic_freq_ghz: Harmonic frequency for markers
    
    Returns:
        Dictionary mapping "metric_signal" to plot file path
    """
    os.makedirs(out_dir, exist_ok=True)
    data = collector.get()
    models = list(data.keys())
    metrics = [
        "insertion_loss",
        "return_loss_left", 
        "return_loss_right",
        "fext",
        "psfext", 
        "next_left",
        "psnext_left",
        "next_right", 
        "psnext_right"
    ]
    signals = [f"DQ{i}" for i in range(8)] + ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
    y_grid = [[[] for _ in range(len(signals))] for _ in range(len(metrics))]
    x_data = None
    for model in models:
        for signal in data[model]:
            if "frequency_ghz" in data[model][signal]:
                x_data = data[model][signal]["frequency_ghz"]
                break
        if x_data:
            break
    for metric_idx, metric in enumerate(metrics):
        for signal_idx, signal in enumerate(signals):
            for model in models:
                if signal in data[model] and metric in data[model][signal]:
                    curve_data = data[model][signal][metric]
                    if curve_data:
                        y_grid[metric_idx][signal_idx].append(curve_data)
    grid_titles = [[f'{metric.replace("_", " ").title()}\n{signal}' for signal in signals] for metric in metrics]
    marker_vlines = [harmonic_freq_ghz, 3*harmonic_freq_ghz] if harmonic_freq_ghz else None
    marker_texts = generate_harmonic_marker_texts(y_grid, x_data, models, harmonic_freq_ghz, len(metrics), len(signals))
    plot_paths = plot_curves_grid_multi(
        y_grid, x_data, models, out_dir, grid_titles=grid_titles,
        xlabel='Freq (GHz)', ylabel='Metric (dB)', 
        img_width=img_width, img_height=img_height,
        nrows=len(metrics), ncols=len(signals), marker_texts=marker_texts, 
        marker_vlines=marker_vlines, legend_loc='best', tight_rect=[0, 0, 0.75, 1]
    )
    metrics_plot_paths = {}
    for metric_idx, metric in enumerate(metrics):
        for signal_idx, signal in enumerate(signals):
            # Save each plot with a unique filename
            plot_key = f"{metric}_{signal}"
            fig_path = os.path.join(out_dir, f"{plot_key}.png")
            # Copy or rename the plot to the unique filename if needed
            orig_path = plot_paths[metric_idx][signal_idx]
            if orig_path and orig_path != fig_path:
                import shutil
                shutil.copyfile(orig_path, fig_path)
            if os.path.exists(fig_path):
                metrics_plot_paths[plot_key] = fig_path
    return metrics_plot_paths

def generate_tdr_grid_plots(tdr_collector, model_names, out_dir, img_width=1200, img_height=800):
    """
    Generate TDR plots using plot_curves_grid_multi from HierarchicalDataCollector data.
    
    Args:
        tdr_collector: HierarchicalDataCollector with TDR data
        model_names: List of model names
        out_dir: Directory to save plots
        img_width, img_height: Image dimensions
    
    Returns:
        Dictionary mapping signal names to plot file paths for left and right sides
    """
    os.makedirs(out_dir, exist_ok=True)
    
    # Signal names in order: DQ0-DQ7, DQS0-, DQS0+, DQS1-, DQS1+
    signal_names = [f"DQ{i}" for i in range(8)] + ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
    
    # Prepare data for left side (row 0) and right side (row 1)
    y_grid = [[[] for _ in range(len(signal_names))] for _ in range(2)]  # 2 rows (left/right), ncols signals
    x_data = None
    
    # Get data from collector
    collector_data = tdr_collector.get()
    
    for col, signal_name in enumerate(signal_names):
        # Get time axis from any model (should be the same for all)
        for model_name in model_names:
            if model_name in collector_data and signal_name in collector_data[model_name]:
                if "time_axis_ns" in collector_data[model_name][signal_name]:
                    x_data = collector_data[model_name][signal_name]["time_axis_ns"]
                    break
        if x_data:
            break
    
    # Extract TDR data for each signal and side
    for col, signal_name in enumerate(signal_names):
        for model_name in model_names:
            if model_name in collector_data and signal_name in collector_data[model_name]:
                signal_data = collector_data[model_name][signal_name]
                
                # Left side TDR (row 0)
                if "tdr_left" in signal_data:
                    y_grid[0][col].append(signal_data["tdr_left"])
                
                # Right side TDR (row 1)
                if "tdr_right" in signal_data:
                    y_grid[1][col].append(signal_data["tdr_right"])
    
    # Generate plots using the existing function
    grid_titles = [
        [f'{signal_name}\n(Left Side)' for signal_name in signal_names],  # Row 0: Left side
        [f'{signal_name}\n(Right Side)' for signal_name in signal_names]  # Row 1: Right side
    ]
    
    plot_paths = plot_curves_grid_multi(
        y_grid, x_data, model_names, out_dir, grid_titles=grid_titles,
        xlabel='Time (ns)', ylabel='TDR Magnitude', 
        img_width=img_width, img_height=img_height,
        nrows=2, ncols=len(signal_names), legend_loc='best', tight_rect=[0, 0, 0.75, 1]
    )
    
    # Convert to the expected format for Excel
    tdr_plot_paths = {}
    
    for col, signal_name in enumerate(signal_names):
        # Left side (row 0)
        if plot_paths[0][col]:
            tdr_plot_paths[f"{signal_name}_left"] = plot_paths[0][col]
        
        # Right side (row 1)
        if plot_paths[1][col]:
            tdr_plot_paths[f"{signal_name}_right"] = plot_paths[1][col]
    
    return tdr_plot_paths

def create_excel_with_metrics_sheet(excel_file, s_matrix_plot_paths, metrics_plot_paths, tdr_plot_paths=None, nports=12, img_width=1200, img_height=800):
    workbook = xlsxwriter.Workbook(excel_file)
    
    # S-matrix sheet
    worksheet1 = workbook.add_worksheet("s-matrix")
    col_width = (img_width / 7.5) * 1.08
    row_height = img_height * 0.75
    for col in range(nports):
        worksheet1.set_column(col, col, col_width)
    for row in range(nports):
        worksheet1.set_row(row, row_height)
    for row in range(nports):
        for col in range(nports):
            cell = xlsxwriter.utility.xl_rowcol_to_cell(row, col)  # type: ignore
            worksheet1.insert_image(cell, s_matrix_plot_paths[row][col],
                                  {'x_scale': 1, 'y_scale': 1, 'object_position': 1})
    
    # Metrics sheet
    worksheet2 = workbook.add_worksheet("metrics of interest")
    metrics = [
        "insertion_loss",
        "return_loss_left", 
        "return_loss_right",
        "fext",
        "psfext", 
        "next_left",
        "psnext_left",
        "next_right", 
        "psnext_right"
    ]
    signals = [f"DQ{i}" for i in range(8)] + ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
    for col in range(len(signals)):
        worksheet2.set_column(col, col, col_width)
    for row in range(len(metrics)):
        worksheet2.set_row(row, row_height)
    if metrics_plot_paths:
        for row, metric in enumerate(metrics):
            for col, signal in enumerate(signals):
                plot_key = f"{metric}_{signal}"
                if plot_key in metrics_plot_paths:
                    cell = xlsxwriter.utility.xl_rowcol_to_cell(row, col)  # type: ignore
                    worksheet2.insert_image(cell, metrics_plot_paths[plot_key],
                                          {'x_scale': 1, 'y_scale': 1, 'object_position': 1})
    
    # TDR sheet
    if tdr_plot_paths:
        worksheet3 = workbook.add_worksheet("TDR")
        signals = [f"DQ{i}" for i in range(8)] + ["DQS0", "DQS1"]
        
        # Set column widths for signals
        for col in range(len(signals)):
            worksheet3.set_column(col, col, col_width)
        
        # Set row heights for left and right side TDR
        worksheet3.set_row(0, row_height)  # Left side TDR
        worksheet3.set_row(1, row_height)  # Right side TDR
        
        # Add TDR plots
        for col, signal in enumerate(signals):
            # Handle DQS signals (combine + and -)
            if signal == "DQS0":
                # Use DQS0- for left side, DQS0+ for right side
                left_plot_key = "DQS0-_left"
                right_plot_key = "DQS0+_right"
            elif signal == "DQS1":
                # Use DQS1- for left side, DQS1+ for right side
                left_plot_key = "DQS1-_left"
                right_plot_key = "DQS1+_right"
            else:
                # DQ signals use left and right side plots
                left_plot_key = f"{signal}_left"
                right_plot_key = f"{signal}_right"
            
            # Left side TDR (row 0)
            if left_plot_key in tdr_plot_paths:
                cell = xlsxwriter.utility.xl_rowcol_to_cell(0, col)  # type: ignore
                worksheet3.insert_image(cell, tdr_plot_paths[left_plot_key],
                                      {'x_scale': 1, 'y_scale': 1, 'object_position': 1})
            
            # Right side TDR (row 1)
            if right_plot_key in tdr_plot_paths:
                cell = xlsxwriter.utility.xl_rowcol_to_cell(1, col)  # type: ignore
                worksheet3.insert_image(cell, tdr_plot_paths[right_plot_key],
                                      {'x_scale': 1, 'y_scale': 1, 'object_position': 1})
    
    workbook.close() 