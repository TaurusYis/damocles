"""
plotting_utils.py
Module for plotting S-parameter curves, metrics, and embedding figures in Excel.
"""
import os
import matplotlib.pyplot as plt
import xlsxwriter
import numpy as np

def plot_s_matrix(s_params, frequencies, out_dir, nports=12, img_width=1200, img_height=800):
    """
    Plot S(row, col) for the first nports and save as images in out_dir.
    Returns a 2D list of image file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_paths = [[None for _ in range(nports)] for _ in range(nports)]
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

def plot_s_matrix_multi(s_params_dict, frequencies, model_names, out_dir, nports=12, img_width=1200, img_height=800):
    """
    For each S(row, col), plot all models' curves together with legends, save as images in out_dir.
    s_params_dict: dict of {model_name: s_params}
    Returns a 2D list of image file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    plot_paths = [[None for _ in range(nports)] for _ in range(nports)]
    dpi = plt.rcParams.get('figure.dpi', 100)
    for row in range(nports):
        for col in range(nports):
            plt.figure(figsize=(img_width/dpi, img_height/dpi))
            for model_name in model_names:
                s_curve = s_params_dict[model_name][:, row, col]
                plt.plot(frequencies / 1e9, 20 * np.log10(np.abs(s_curve)), lw=2, label=model_name)
            plt.xlabel('Freq (GHz)', fontsize=14)
            plt.ylabel(f'S{row+1},{col+1} (dB)', fontsize=14)
            plt.title(f'S{row+1},{col+1}', fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.tight_layout()
            fig_path = os.path.join(out_dir, f"S{row+1}_{col+1}.png")
            plt.savefig(fig_path)
            plt.close()
            plot_paths[row][col] = fig_path
    return plot_paths

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
            cell = xlsxwriter.utility.xl_rowcol_to_cell(row, col)
            worksheet.insert_image(cell, plot_paths[row][col],
                                  {'x_scale': 1, 'y_scale': 1, 'object_position': 1})
    worksheet2 = workbook.add_worksheet("metrics of interest")
    worksheet2.write(0, 0, "(Plots for metrics of interest will be added here)")
    workbook.close() 