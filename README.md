# DDR S-Parameter Analysis Tool

A modular Python tool for generating, processing, and visualizing DDR channel S-parameters (Touchstone files), with Excel reporting.

## Features

- **Touchstone Generation**: Create realistic example S-parameter files for DDR channels.
- **Touchstone Processing**: Read Touchstone files, extract S-parameters, and calculate metrics (insertion loss, return loss, crosstalk, TDR, etc.).
- **Plotting & Excel Embedding**: Plot S-parameter curves, save figures, and embed them in a well-organized Excel report.

## Project Structure

```
.
├── main.py                  # Main workflow script
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
   - Create an Excel report (`ddr_report.xlsx`) with all S-matrix plots embedded in a 12x12 grid

3. **Open `ddr_report.xlsx`** to view the results.

## How It Works

- **Touchstone Generation**: `touchstone_generator.py` provides functions to create realistic multi-port S-parameter files for DDR channel analysis.
- **Touchstone Processing**: `touchstone_processor.py` reads Touchstone files and calculates metrics of interest.
- **Plotting & Excel**: `plotting_utils.py` handles all plotting and Excel embedding, ensuring a visually appealing report.
- **Orchestration**: `main.py` ties everything together for a seamless workflow.

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