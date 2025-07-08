#!/usr/bin/env python3
"""
Example: Using the Simplified API
Demonstrates how to generate a DDR report with minimal data preparation.
"""

import numpy as np
from main import run_report_from_external_data_simplified

def create_minimal_example_data():
    """
    Create minimal example data to demonstrate the simplified API.
    In practice, you would load this data from your own sources.
    """
    # Example model names
    model_names = ["Design_A", "Design_B"]
    
    # Create minimal S-matrix data (24-port system)
    s_matrix_data = {}
    for model_name in model_names:
        s_matrix_data[model_name] = {}
        
        # Create frequency axis (1 MHz to 20 GHz, 1001 points)
        frequencies = np.logspace(6, 10, 1001)  # 1 MHz to 10 GHz
        
        # Create simple S-parameter data for each port pair
        for row in range(1, 25):  # 24 ports
            for col in range(1, 25):
                port_pair = f"({row},{col})"
                
                if row == col:
                    # Return loss (diagonal elements)
                    s_db = -15 - 2 * (frequencies / 1e9)  # Frequency-dependent return loss
                else:
                    # Insertion loss (off-diagonal elements)
                    s_db = -30 - 3 * (frequencies / 1e9) - 2 * abs(row - col)  # Distance-dependent
                
                # Add some variation between models
                if model_name == "Design_B":
                    s_db += 2  # Slightly better performance
                
                s_matrix_data[model_name][port_pair] = {
                    "s_parameter_db": s_db.tolist(),
                    "frequency_ghz": (frequencies / 1e9).tolist()
                }
    
    # Create minimal TDR data
    tdr_data = {}
    for model_name in model_names:
        tdr_data[model_name] = {}
        
        # Time axis (0 to 10 ns, 1000 points)
        time_axis = np.linspace(0, 10, 1000)
        
        # Signal names: DQ0-DQ7, DQS0-, DQS0+, DQS1-, DQS1+
        signal_names = [f"DQ{i}" for i in range(8)] + ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
        
        for signal_name in signal_names:
            # Create simple TDR response (Gaussian pulse)
            t = time_axis
            if "DQ" in signal_name:
                # DQ signals: peak at 2 ns
                tdr_left = np.exp(-((t - 2) / 0.5)**2)
                tdr_right = np.exp(-((t - 8) / 0.5)**2)
            else:
                # DQS signals: peak at 3 ns
                tdr_left = np.exp(-((t - 3) / 0.3)**2)
                tdr_right = np.exp(-((t - 7) / 0.3)**2)
            
            # Add variation between models
            if model_name == "Design_B":
                tdr_left *= 1.2
                tdr_right *= 1.2
            
            tdr_data[model_name][signal_name] = {
                "tdr_left": tdr_left.tolist(),
                "tdr_right": tdr_right.tolist(),
                "time_axis_ns": (time_axis * 1e9).tolist()
            }
    
    return s_matrix_data, tdr_data, model_names

def main():
    """
    Demonstrate the simplified API with minimal example data.
    """
    print("DDR S-Parameter Analysis Tool - Simplified API Example")
    print("=" * 60)
    
    # Create minimal example data
    print("Creating minimal example data...")
    s_matrix_data, tdr_data, model_names = create_minimal_example_data()
    print(f"✓ Created data for {len(model_names)} models: {model_names}")
    
    # Generate report using simplified API
    print("\nGenerating report using simplified API...")
    print("(Metrics will be calculated automatically from S-matrix data)")
    
    run_report_from_external_data_simplified(
        s_matrix_data,
        tdr_data,
        model_names,
        output_prefix="simplified_example"
    )
    
    print("\nExample complete!")
    print("Check the generated files:")
    print("  - simplified_example_report.xlsx")
    print("  - figures/simplified_example_*/")

if __name__ == "__main__":
    main() 