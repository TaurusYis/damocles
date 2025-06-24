import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.interpolate import interp1d
import os
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import skrf, fallback to manual parsing if not available
try:
    import skrf as rf
    SKRF_AVAILABLE = True
except ImportError:
    SKRF_AVAILABLE = False
    print("Warning: skrf not available, using manual Touchstone parsing")

class DDRSParameterAnalyzer:
    """
    DDR S-Parameter Analyzer for comparing different channel models
    """
    
    def __init__(self):
        self.models = {}  # Dictionary to store different model data
        self.frequencies = None
        self.port_names = []
        
    def read_touchstone_file(self, filepath: str, model_name: str) -> bool:
        """
        Read Touchstone file and store S-parameters for a model
        
        Args:
            filepath: Path to the Touchstone file
            model_name: Name to identify this model (e.g., 'CPU_Socket_v1')
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if SKRF_AVAILABLE:
                # Use skrf for robust Touchstone parsing
                network = rf.Network(filepath)
                self.models[model_name] = {
                    'frequencies': network.f,
                    's_params': network.s,
                    'port_names': [f'Port_{i+1}' for i in range(network.nports)]
                }
            else:
                # Manual Touchstone parsing
                self._parse_touchstone_manual(filepath, model_name)
                
            # Update global frequency and port info if first model
            if self.frequencies is None:
                self.frequencies = self.models[model_name]['frequencies']
                self.port_names = self.models[model_name]['port_names']
                
            print(f"Successfully loaded model: {model_name}")
            print(f"  - Frequency range: {self.frequencies[0]/1e9:.2f} - {self.frequencies[-1]/1e9:.2f} GHz")
            print(f"  - Number of ports: {len(self.port_names)}")
            return True
            
        except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")
            return False
    
    def _parse_touchstone_manual(self, filepath: str, model_name: str):
        """Manual Touchstone file parser as fallback"""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        header_line = lines[0].strip()
        # Extract frequency unit and format
        freq_unit = 'GHz' if 'GHz' in header_line else 'Hz'
        format_type = 'RI' if 'RI' in header_line else 'MA'
        
        # Parse data
        data_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith('!'):
                data_lines.append(line)
        
        # Convert to numpy arrays
        data = np.array([[float(x) for x in line.split()] for line in data_lines])
        
        # Extract frequencies and S-parameters
        frequencies = data[:, 0]
        if freq_unit == 'GHz':
            frequencies *= 1e9
        
        s_params = data[:, 1:]
        
        # Convert to complex format if needed
        if format_type == 'RI':
            # Real/Imaginary format
            s_params_complex = s_params[:, 0::2] + 1j * s_params[:, 1::2]
        else:
            # Magnitude/Phase format
            mag = s_params[:, 0::2]
            phase = s_params[:, 1::2] * np.pi / 180
            s_params_complex = mag * np.exp(1j * phase)
        
        # Reshape to (freq_points, nports, nports)
        nports = int(np.sqrt(s_params_complex.shape[1]))
        s_params_reshaped = s_params_complex.reshape(-1, nports, nports)
        
        self.models[model_name] = {
            'frequencies': frequencies,
            's_params': s_params_reshaped,
            'port_names': [f'Port_{i+1}' for i in range(nports)]
        }
    
    def define_ddr_ports(self, dq_ports: List[int], dqs_ports: List[int], 
                        dqs_pairs: List[Tuple[int, int]]):
        """
        Define DDR port mapping for DQ and DQS signals
        
        Args:
            dq_ports: List of DQ port numbers (1-based indexing)
            dqs_ports: List of DQS port numbers (1-based indexing)
            dqs_pairs: List of tuples for DQS differential pairs [(pos, neg), ...]
        """
        self.dq_ports = dq_ports
        self.dqs_ports = dqs_ports
        self.dqs_pairs = dqs_pairs
        
        print(f"Defined DDR ports:")
        print(f"  - DQ ports: {dq_ports}")
        print(f"  - DQS ports: {dqs_ports}")
        print(f"  - DQS pairs: {dqs_pairs}")
    
    def calculate_insertion_loss(self, model_name: str) -> Dict[str, np.ndarray]:
        """
        Calculate insertion loss for DQ and DQS signals
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with insertion loss data for each signal
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        frequencies = model_data['frequencies']
        
        il_results = {}
        
        # Calculate insertion loss for DQ signals
        for dq_port in self.dq_ports:
            # Assuming DQ ports are single-ended
            il_db = 20 * np.log10(np.abs(s_params[:, dq_port-1, dq_port-1]))
            il_results[f'DQ_{dq_port}'] = il_db
        
        # Calculate insertion loss for DQS differential signals
        for i, (pos_port, neg_port) in enumerate(self.dqs_pairs):
            # Differential insertion loss
            s_pos = s_params[:, pos_port-1, pos_port-1]
            s_neg = s_params[:, neg_port-1, neg_port-1]
            s_diff = s_pos - s_neg  # Differential S-parameter
            il_db = 20 * np.log10(np.abs(s_diff))
            il_results[f'DQS_{i+1}_diff'] = il_db
        
        return il_results
    
    def calculate_return_loss(self, model_name: str) -> Dict[str, np.ndarray]:
        """
        Calculate return loss for all ports
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with return loss data for each port
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        
        rl_results = {}
        
        for i in range(s_params.shape[1]):
            rl_db = 20 * np.log10(np.abs(s_params[:, i, i]))
            rl_results[f'Port_{i+1}'] = rl_db
        
        return rl_results
    
    def calculate_tdr(self, model_name: str, time_window: float = 10e-9) -> Dict[str, np.ndarray]:
        """
        Calculate Time Domain Reflectometry (TDR) for all ports
        
        Args:
            model_name: Name of the model to analyze
            time_window: Time window for TDR calculation (seconds)
            
        Returns:
            Dictionary with TDR data for each port
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        frequencies = model_data['frequencies']
        
        # Create time axis
        dt = time_window / (2 * len(frequencies))
        time_axis = np.arange(0, time_window, dt)
        
        tdr_results = {}
        
        for i in range(s_params.shape[1]):
            # Get reflection coefficient
            s11 = s_params[:, i, i]
            
            # Pad with zeros for better time resolution
            s11_padded = np.pad(s11, (0, len(s11)), mode='constant')
            
            # Apply window to reduce ringing
            window = np.hanning(len(s11_padded))
            s11_windowed = s11_padded * window
            
            # Inverse FFT to get time domain response
            tdr_time = np.fft.ifft(s11_windowed)
            tdr_magnitude = np.abs(tdr_time)
            
            # Truncate to time window
            n_points = min(len(tdr_magnitude), len(time_axis))
            tdr_results[f'Port_{i+1}'] = tdr_magnitude[:n_points]
        
        tdr_results['time_axis'] = time_axis[:n_points]
        return tdr_results
    
    def calculate_crosstalk(self, model_name: str) -> Dict[str, np.ndarray]:
        """
        Calculate FEXT and NEXT for DQ signals
        
        Args:
            model_name: Name of the model to analyze
            
        Returns:
            Dictionary with crosstalk data
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        
        crosstalk_results = {}
        
        # Calculate FEXT (Far-End Crosstalk)
        for i, dq1 in enumerate(self.dq_ports):
            for j, dq2 in enumerate(self.dq_ports):
                if i != j:
                    # FEXT: S21 from aggressor to victim
                    fext_db = 20 * np.log10(np.abs(s_params[:, dq2-1, dq1-1]))
                    crosstalk_results[f'FEXT_DQ{dq1}_to_DQ{dq2}'] = fext_db
        
        # Calculate NEXT (Near-End Crosstalk)
        for i, dq1 in enumerate(self.dq_ports):
            for j, dq2 in enumerate(self.dq_ports):
                if i != j:
                    # NEXT: S12 from aggressor to victim
                    next_db = 20 * np.log10(np.abs(s_params[:, dq1-1, dq2-1]))
                    crosstalk_results[f'NEXT_DQ{dq1}_to_DQ{dq2}'] = next_db
        
        return crosstalk_results
    
    def plot_metric_comparison(self, metric_type: str, save_path: str = None):
        """
        Plot comparison of a specific metric across all models
        
        Args:
            metric_type: Type of metric ('insertion_loss', 'return_loss', 'tdr', 'crosstalk')
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        if metric_type == 'insertion_loss':
            self._plot_insertion_loss_comparison()
        elif metric_type == 'return_loss':
            self._plot_return_loss_comparison()
        elif metric_type == 'tdr':
            self._plot_tdr_comparison()
        elif metric_type == 'crosstalk':
            self._plot_crosstalk_comparison()
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def _plot_insertion_loss_comparison(self):
        """Plot insertion loss comparison across models"""
        plt.subplot(2, 1, 1)
        for model_name in self.models.keys():
            il_data = self.calculate_insertion_loss(model_name)
            for signal, il_values in il_data.items():
                if signal.startswith('DQ'):
                    plt.plot(self.frequencies/1e9, il_values, 
                            label=f'{model_name}_{signal}', linewidth=2)
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Insertion Loss (dB)')
        plt.title('DQ Insertion Loss Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        for model_name in self.models.keys():
            il_data = self.calculate_insertion_loss(model_name)
            for signal, il_values in il_data.items():
                if signal.startswith('DQS'):
                    plt.plot(self.frequencies/1e9, il_values, 
                            label=f'{model_name}_{signal}', linewidth=2)
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Insertion Loss (dB)')
        plt.title('DQS Differential Insertion Loss Comparison')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
    
    def _plot_return_loss_comparison(self):
        """Plot return loss comparison across models"""
        n_models = len(self.models)
        n_cols = 3
        n_rows = (len(self.port_names) + n_cols - 1) // n_cols
        
        for i, port_name in enumerate(self.port_names):
            plt.subplot(n_rows, n_cols, i + 1)
            
            for model_name in self.models.keys():
                rl_data = self.calculate_return_loss(model_name)
                rl_values = rl_data[port_name]
                plt.plot(self.frequencies/1e9, rl_values, 
                        label=model_name, linewidth=2)
            
            plt.xlabel('Frequency (GHz)')
            plt.ylabel('Return Loss (dB)')
            plt.title(f'{port_name} Return Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    def _plot_tdr_comparison(self):
        """Plot TDR comparison across models"""
        n_models = len(self.models)
        n_cols = 3
        n_rows = (len(self.port_names) + n_cols - 1) // n_cols
        
        for i, port_name in enumerate(self.port_names):
            plt.subplot(n_rows, n_cols, i + 1)
            
            for model_name in self.models.keys():
                tdr_data = self.calculate_tdr(model_name)
                tdr_values = tdr_data[port_name]
                time_axis = tdr_data['time_axis']
                plt.plot(time_axis*1e9, tdr_values, 
                        label=model_name, linewidth=2)
            
            plt.xlabel('Time (ns)')
            plt.ylabel('TDR Magnitude')
            plt.title(f'{port_name} TDR')
            plt.legend()
            plt.grid(True, alpha=0.3)
    
    def _plot_crosstalk_comparison(self):
        """Plot crosstalk comparison across models"""
        # Focus on worst-case crosstalk
        plt.subplot(2, 1, 1)
        for model_name in self.models.keys():
            xtalk_data = self.calculate_crosstalk(model_name)
            
            # Find worst-case FEXT
            fext_values = []
            for key, values in xtalk_data.items():
                if key.startswith('FEXT'):
                    fext_values.extend(values)
            
            if fext_values:
                worst_fext = np.max(fext_values)
                plt.plot(self.frequencies/1e9, worst_fext, 
                        label=f'{model_name}_Worst_FEXT', linewidth=2)
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('FEXT (dB)')
        plt.title('Worst-Case FEXT Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        for model_name in self.models.keys():
            xtalk_data = self.calculate_crosstalk(model_name)
            
            # Find worst-case NEXT
            next_values = []
            for key, values in xtalk_data.items():
                if key.startswith('NEXT'):
                    next_values.extend(values)
            
            if next_values:
                worst_next = np.max(next_values)
                plt.plot(self.frequencies/1e9, worst_next, 
                        label=f'{model_name}_Worst_NEXT', linewidth=2)
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('NEXT (dB)')
        plt.title('Worst-Case NEXT Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    def export_to_excel(self, output_file: str):
        """
        Export all analysis results to Excel file
        
        Args:
            output_file: Path to the output Excel file
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Create summary sheet
            summary_data = []
            for model_name in self.models.keys():
                model_data = self.models[model_name]
                summary_data.append({
                    'Model': model_name,
                    'Frequency Range (GHz)': f"{model_data['frequencies'][0]/1e9:.2f} - {model_data['frequencies'][-1]/1e9:.2f}",
                    'Number of Ports': len(model_data['port_names']),
                    'Number of Frequency Points': len(model_data['frequencies'])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create sheets for each metric
            for model_name in self.models.keys():
                # Insertion Loss
                il_data = self.calculate_insertion_loss(model_name)
                il_df = pd.DataFrame(il_data)
                il_df.insert(0, 'Frequency_GHz', self.frequencies/1e9)
                il_df.to_excel(writer, sheet_name=f'{model_name}_Insertion_Loss', index=False)
                
                # Return Loss
                rl_data = self.calculate_return_loss(model_name)
                rl_df = pd.DataFrame(rl_data)
                rl_df.insert(0, 'Frequency_GHz', self.frequencies/1e9)
                rl_df.to_excel(writer, sheet_name=f'{model_name}_Return_Loss', index=False)
                
                # Crosstalk
                xtalk_data = self.calculate_crosstalk(model_name)
                xtalk_df = pd.DataFrame(xtalk_data)
                xtalk_df.insert(0, 'Frequency_GHz', self.frequencies/1e9)
                xtalk_df.to_excel(writer, sheet_name=f'{model_name}_Crosstalk', index=False)
                
                # TDR
                tdr_data = self.calculate_tdr(model_name)
                tdr_df = pd.DataFrame({k: v for k, v in tdr_data.items() if k != 'time_axis'})
                tdr_df.insert(0, 'Time_ns', tdr_data['time_axis']*1e9)
                tdr_df.to_excel(writer, sheet_name=f'{model_name}_TDR', index=False)
        
        print(f"Analysis results exported to: {output_file}")
    
    def generate_comparison_report(self, output_file: str):
        """
        Generate a comprehensive comparison report
        
        Args:
            output_file: Path to the output Excel file
        """
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Create comparison sheets for each metric
            metrics = ['insertion_loss', 'return_loss', 'crosstalk']
            
            for metric in metrics:
                comparison_data = {}
                
                if metric == 'insertion_loss':
                    for model_name in self.models.keys():
                        il_data = self.calculate_insertion_loss(model_name)
                        for signal, values in il_data.items():
                            comparison_data[f'{model_name}_{signal}'] = values
                
                elif metric == 'return_loss':
                    for model_name in self.models.keys():
                        rl_data = self.calculate_return_loss(model_name)
                        for port, values in rl_data.items():
                            comparison_data[f'{model_name}_{port}'] = values
                
                elif metric == 'crosstalk':
                    for model_name in self.models.keys():
                        xtalk_data = self.calculate_crosstalk(model_name)
                        for xtalk_type, values in xtalk_data.items():
                            comparison_data[f'{model_name}_{xtalk_type}'] = values
                
                # Add frequency column
                comparison_data['Frequency_GHz'] = self.frequencies/1e9
                
                # Create DataFrame and save
                comparison_df = pd.DataFrame(comparison_data)
                comparison_df.to_excel(writer, sheet_name=f'{metric}_comparison', index=False)
        
        print(f"Comparison report exported to: {output_file}") 