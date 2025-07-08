# DDR S-Parameter Analysis Tool

A modular Python tool for generating, processing, and visualizing DDR channel S-parameters (Touchstone files), with Excel reporting.

## Features

- **Touchstone Generation**: Create realistic example S-parameter files for DDR channels.
- **Touchstone Processing**: Read Touchstone files, extract S-parameters, and calculate metrics (insertion loss, return loss, crosstalk, TDR, etc.).
- **Plotting & Excel Embedding**: Plot S-parameter curves, save figures, and embed them in a well-organized Excel report.
- **Easy Integration**: Feed your own processed data from other programs and generate reports directly (see below).

## Project Structure

```
.
├── main.py                  # Main workflow script (includes simplified API)
├── example_simplified_api.py # Example of using the simplified API
├── touchstone_generator.py  # Generate example Touchstone files
├── touchstone_processor.py  # Read/process Touchstone files, calculate metrics
├── plotting_utils.py        # Plotting and Excel embedding utilities
├── sparams/                 # Folder for generated Touchstone files
├── figures/                 # Folder for generated figures
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main workflow:**
   ```bash
   python main.py
   ```

   This will:
   - Generate example Touchstone files in `sparams/`
   - Read and process the files
   - Plot S-matrix figures and save them in `figures/`
   - Create an Excel report (`ddr_report.xlsx`) with all S-matrix, metrics, and TDR plots

3. **Open `ddr_report.xlsx`** to view the results.

## How It Works

- **Touchstone Generation**: `touchstone_generator.py` provides functions to create realistic multi-port S-parameter files for DDR channel analysis.
- **Touchstone Processing**: `touchstone_processor.py` reads Touchstone files and calculates metrics of interest.
- **Plotting & Excel**: `plotting_utils.py` handles all plotting and Excel embedding, ensuring a visually appealing report.
- **Orchestration**: `main.py` ties everything together for a seamless workflow.

## Integrating with Other Programs

You can use your own code to load/process Touchstone files or generate S-parameter/metrics/TDR data, and then feed that data directly into the reporting workflow. This is useful if you already have a custom loader or want to connect with other tools.

### **Minimal Example: Generate a Report from Your Own Data**

#### **Simplified API (Recommended)**
```python
from main import run_report_from_external_data_simplified

# Prepare your data as dictionaries (see below for structure)
s_matrix_data = { ... }   # dict: model_name -> s-matrix data
tdr_data = { ... }        # dict: model_name -> TDR data
model_names = [ ... ]     # list of model names (order for plotting)

# Call the simplified API - metrics are calculated automatically from S-matrix data
run_report_from_external_data_simplified(
    s_matrix_data,
    tdr_data,
    model_names,
    output_prefix="my_external_analysis"  # Optional: prefix for output files
)
```

#### **Full API (if you want to provide pre-calculated metrics)**
```python
from main import run_report_from_external_data

# Prepare your data as dictionaries
s_matrix_data = { ... }   # dict: model_name -> s-matrix data
metrics_data = { ... }    # dict: model_name -> metrics data (optional)
tdr_data = { ... }        # dict: model_name -> TDR data
model_names = [ ... ]     # list of model names (order for plotting)

# Call the full API
run_report_from_external_data(
    s_matrix_data,
    metrics_data,
    tdr_data,
    model_names,
    output_prefix="my_external_analysis"
)
```

#### **Data Structure Example**
- **S-matrix data**: `s_matrix_data[model_name]["(row,col)"]["s_parameter_db"]` (list of dB values)
- **TDR data**: `tdr_data[model_name][signal]["tdr_left"]`, `tdr_right`, `time_axis_ns`, etc.
- **Metrics data**: Automatically calculated from S-matrix data using the same algorithms as the internal workflow
- See the code in `s_parameter_processor.py` for details, or run the internal workflow once and inspect the generated `*_data.pkl` files for examples.

#### **Tip:**
You can use the `HierarchicalDataCollector` and `DataOrganizer` utilities to help organize your data if needed.

#### **Example Script:**
Run `python example_simplified_api.py` to see a complete example of using the simplified API with minimal data preparation.

## Customization

- To analyze your own Touchstone files, place them in `sparams/` and update `MODEL_NAME` in `main.py`.
- To change the number of ports or figure size, adjust the parameters in `main.py` and `plotting_utils.py`.
- To add new metrics or plots, extend `touchstone_processor.py` and `plotting_utils.py` as needed.

## Requirements

- Python 3.7+
- numpy
- matplotlib
- xlsxwriter
- scikit-rf (optional, for robust Touchstone parsing)

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## License

This tool is provided as-is for educational and research purposes.

## Contributing

Contributions and suggestions are welcome! Open an issue or submit a pull request to improve the tool. 