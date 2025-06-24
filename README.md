# DDR S-Parameter Analysis Tool

A comprehensive automation tool for analyzing DDR channel S-parameters from Touchstone files, with built-in comparison capabilities for different channel models.

## Features

- **S-Parameter Reading**: Supports Touchstone (.s2p, .s3p, etc.) files with automatic format detection
- **DDR-Specific Analysis**: 
  - Insertion loss for 8 DQ signals
  - Differential insertion loss for 2 DQS pairs
  - Return loss for all 24 ports
  - Time Domain Reflectometry (TDR) for all ports
  - Far-End Crosstalk (FEXT) and Near-End Crosstalk (NEXT)
- **Model Comparison**: Side-by-side comparison of different channel models (e.g., different CPU socket versions)
- **Visualization**: Automatic generation of comparison plots
- **Excel Export**: Comprehensive Excel reports with multiple sheets for detailed analysis

## Installation

1. Clone or download this repository
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from ddr_s_parameter_analyzer import DDRSParameterAnalyzer

# Create analyzer instance
analyzer = DDRSParameterAnalyzer()

# Load your Touchstone files
analyzer.read_touchstone_file("cpu_socket_v1.s2p", "CPU_Socket_v1")
analyzer.read_touchstone_file("cpu_socket_v2.s2p", "CPU_Socket_v2")

# Define DDR port mapping
dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]  # DQ0-DQ7
dqs_ports = [9, 10, 11, 12]  # DQS0+, DQS0-, DQS1+, DQS1-
dqs_pairs = [(9, 10), (11, 12)]  # DQS0 and DQS1 differential pairs

analyzer.define_ddr_ports(dq_ports, dqs_ports, dqs_pairs)

# Generate comparison plots
analyzer.plot_metric_comparison('insertion_loss')
analyzer.plot_metric_comparison('return_loss')
analyzer.plot_metric_comparison('tdr')
analyzer.plot_metric_comparison('crosstalk')

# Export to Excel
analyzer.export_to_excel("ddr_analysis_results.xlsx")
analyzer.generate_comparison_report("ddr_comparison_report.xlsx")
```

### Run Demo

To see the tool in action with sample data:

```bash
python main.py
```

This will:
1. Create sample Touchstone files
2. Load and analyze the data
3. Generate comparison plots
4. Export Excel reports

## Input File Format

The tool expects Touchstone files (.s2p, .s3p, etc.) with the following format:

```
# GHz S RI R 50
! Frequency(GHz) S(1,1) S(1,2) ... S(1,N) S(2,1) ... S(N,N)
1.000000e-03 1.234567e-01 2.345678e-02 ... 1.234567e-01 2.345678e-02 ...
2.000000e-03 1.234567e-01 2.345678e-02 ... 1.234567e-01 2.345678e-02 ...
...
```

## DDR Port Mapping

The tool requires you to define the DDR port mapping:

- **DQ Ports**: Single-ended data signals (typically 8 signals)
- **DQS Ports**: Differential strobe signals (typically 2 pairs)
- **DQS Pairs**: Define which ports form differential pairs

Example for a 24-port system:
```python
dq_ports = [1, 2, 3, 4, 5, 6, 7, 8]  # DQ0-DQ7
dqs_ports = [9, 10, 11, 12]  # DQS0+, DQS0-, DQS1+, DQS1-
dqs_pairs = [(9, 10), (11, 12)]  # DQS0 and DQS1 differential pairs
```

## Output Files

### Plots
- `insertion_loss_comparison.png`: DQ and DQS insertion loss comparison
- `return_loss_comparison.png`: Return loss for all ports
- `tdr_comparison.png`: Time Domain Reflectometry for all ports
- `crosstalk_comparison.png`: FEXT and NEXT comparison

### Excel Reports
- `ddr_analysis_results.xlsx`: Detailed analysis with separate sheets for each model
- `ddr_comparison_report.xlsx`: Side-by-side comparison of all models

## Metrics Calculated

### Insertion Loss
- **DQ Signals**: Single-ended insertion loss for each DQ signal
- **DQS Signals**: Differential insertion loss for DQS pairs

### Return Loss
- Reflection coefficient magnitude for all ports
- Useful for impedance matching analysis

### Time Domain Reflectometry (TDR)
- Time-domain reflection response
- Helps identify impedance discontinuities and their locations

### Crosstalk
- **FEXT (Far-End Crosstalk)**: Crosstalk at the far end of the channel
- **NEXT (Near-End Crosstalk)**: Crosstalk at the near end of the channel

## Advanced Usage

### Custom Analysis

```python
# Calculate specific metrics for a model
il_data = analyzer.calculate_insertion_loss("CPU_Socket_v1")
rl_data = analyzer.calculate_return_loss("CPU_Socket_v1")
tdr_data = analyzer.calculate_tdr("CPU_Socket_v1")
xtalk_data = analyzer.calculate_crosstalk("CPU_Socket_v1")

# Access specific signals
dq1_il = il_data['DQ_1']
dqs1_diff_il = il_data['DQS_1_diff']
port1_rl = rl_data['Port_1']
```

### Custom Plotting

```python
import matplotlib.pyplot as plt

# Get data for custom plotting
il_data = analyzer.calculate_insertion_loss("CPU_Socket_v1")
frequencies = analyzer.frequencies

plt.figure(figsize=(10, 6))
for signal, values in il_data.items():
    if signal.startswith('DQ'):
        plt.plot(frequencies/1e9, values, label=signal)
plt.xlabel('Frequency (GHz)')
plt.ylabel('Insertion Loss (dB)')
plt.title('DQ Insertion Loss')
plt.legend()
plt.grid(True)
plt.show()
```

## Troubleshooting

### Common Issues

1. **Import Error for skrf**: Install with `pip install scikit-rf`
2. **Touchstone Format**: Ensure your file follows the standard Touchstone format
3. **Port Mapping**: Verify that your port numbers match the actual S-parameter file

### File Format Support

The tool supports:
- Real/Imaginary (RI) format
- Magnitude/Phase (MA) format
- Different frequency units (Hz, kHz, MHz, GHz)
- Different reference impedances

## Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing functions
- `matplotlib`: Plotting
- `pandas`: Data manipulation
- `openpyxl`: Excel file handling
- `scikit-rf`: RF network analysis (optional, falls back to manual parsing)

## License

This tool is provided as-is for educational and research purposes.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the tool. 