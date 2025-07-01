"""
s_parameter_processor.py
Module for post-processing S-parameters and calculating various metrics.
This module is designed to be easily replaceable with other analysis methods.
"""
import numpy as np
from typing import Dict, List, Tuple, Any
from tdr_analyzer import calculate_tdr_from_s_params

class SParameterPostProcessor:
    """
    Post-processor for S-parameters that calculates various metrics and returns
    data organized for HierarchicalDataCollector.
    """
    
    def __init__(self):
        self.dq_left_ports = list(range(1, 9))   # DQ0-DQ7 left
        self.dq_right_ports = list(range(13, 21)) # DQ0-DQ7 right
        self.dqs_left_ports = {"DQS0-": 9, "DQS0+": 10, "DQS1-": 11, "DQS1+": 12}
        self.dqs_right_ports = {"DQS0-": 21, "DQS0+": 22, "DQS1-": 23, "DQS1+": 24}
        self.dq_names = [f"DQ{i}" for i in range(8)]
        self.dqs_names = ["DQS0-", "DQS0+", "DQS1-", "DQS1+"]
        self.signal_names = self.dq_names + self.dqs_names
    
    def process_s_matrix_data(self, s_params: np.ndarray, frequencies: np.ndarray, 
                            model_name: str) -> Dict[str, Any]:
        """
        Process S-matrix data and return organized data for collector.
        
        Args:
            s_params: S-parameters array (nfreq, nports, nports)
            frequencies: Frequency array
            model_name: Name of the model
            
        Returns:
            Dictionary with organized S-matrix data
        """
        nports = s_params.shape[1]
        data = {}
        
        for row in range(nports):
            for col in range(nports):
                s_curve = s_params[:, row, col]
                s_db = 20 * np.log10(np.abs(s_curve))
                
                data[f"({row+1},{col+1})"] = {
                    "s_parameter_db": s_db.tolist(),
                    "frequency_ghz": (frequencies/1e9).tolist()
                }
        
        return data
    
    def process_ddr_metrics(self, s_params: np.ndarray, frequencies: np.ndarray, 
                          model_name: str) -> Dict[str, Any]:
        """
        Process DDR-specific metrics and return organized data for collector.
        
        Args:
            s_params: S-parameters array (nfreq, nports, nports)
            frequencies: Frequency array
            model_name: Name of the model
            
        Returns:
            Dictionary with organized DDR metrics data
        """
        data = {}
        
        # Process DQ signals (left side)
        for i, dq_port in enumerate(self.dq_left_ports):
            signal_name = f"DQ{i}"
            port_idx = dq_port - 1
            
            # Basic S-parameter metrics
            s_curve = s_params[:, port_idx, port_idx]
            insertion_loss = -20 * np.log10(np.abs(s_curve))
            return_loss_left = -20 * np.log10(np.abs(s_curve))
            return_loss_right = return_loss_left  # Simplified for now
            
            data[signal_name] = {
                "insertion_loss": insertion_loss.tolist(),
                "return_loss_left": return_loss_left.tolist(),
                "return_loss_right": return_loss_right.tolist(),
                "frequency_ghz": (frequencies/1e9).tolist()
            }
            
            # Calculate FEXT/NEXT metrics
            fext_values = []
            next_values = []
            
            for j, other_port in enumerate(self.dq_left_ports):
                if j != i:
                    other_idx = other_port - 1
                    
                    # FEXT (Far-End Crosstalk)
                    fext_curve = s_params[:, other_idx, port_idx]
                    fext_db = 20 * np.log10(np.abs(fext_curve))
                    fext_values.append(fext_db)
                    
                    # NEXT (Near-End Crosstalk)
                    next_curve = s_params[:, port_idx, other_idx]
                    next_db = 20 * np.log10(np.abs(next_curve))
                    next_values.append(next_db)
            
            if fext_values:
                worst_fext = np.maximum.reduce(fext_values)
                psfext = 10 * np.log10(np.sum([10**(fext/10) for fext in fext_values], axis=0))
                data[signal_name].update({
                    "fext": worst_fext.tolist(),
                    "psfext": psfext.tolist()
                })
            
            if next_values:
                worst_next = np.maximum.reduce(next_values)
                psnext = 10 * np.log10(np.sum([10**(next/10) for next in next_values], axis=0))
                data[signal_name].update({
                    "next_left": worst_next.tolist(),
                    "next_right": worst_next.tolist(),
                    "psnext_left": psnext.tolist(),
                    "psnext_right": psnext.tolist()
                })
        
        # Process DQS signals (differential)
        for dqs_name, dqs_port in self.dqs_left_ports.items():
            port_idx = dqs_port - 1
            
            # Basic S-parameter metrics
            s_curve = s_params[:, port_idx, port_idx]
            insertion_loss = -20 * np.log10(np.abs(s_curve))
            return_loss_left = -20 * np.log10(np.abs(s_curve))
            return_loss_right = return_loss_left  # Simplified for now
            
            data[dqs_name] = {
                "insertion_loss": insertion_loss.tolist(),
                "return_loss_left": return_loss_left.tolist(),
                "return_loss_right": return_loss_right.tolist(),
                "frequency_ghz": (frequencies/1e9).tolist()
            }
            
            # Placeholder for differential crosstalk (simplified)
            data[dqs_name].update({
                "fext": (-40 * np.ones_like(frequencies)).tolist(),
                "psfext": (-40 * np.ones_like(frequencies)).tolist(),
                "next_left": (-40 * np.ones_like(frequencies)).tolist(),
                "next_right": (-40 * np.ones_like(frequencies)).tolist(),
                "psnext_left": (-40 * np.ones_like(frequencies)).tolist(),
                "psnext_right": (-40 * np.ones_like(frequencies)).tolist()
            })
        
        return data
    
    def process_tdr_data(self, s_params: np.ndarray, frequencies: np.ndarray, 
                        model_name: str) -> Dict[str, Any]:
        """
        Process TDR data and return organized data for collector.
        
        Args:
            s_params: S-parameters array (nfreq, nports, nports)
            frequencies: Frequency array
            model_name: Name of the model
            
        Returns:
            Dictionary with organized TDR data
        """
        # Calculate TDR for all ports
        tdr_all_ports = calculate_tdr_from_s_params(s_params, frequencies)
        time_axis = tdr_all_ports['time_axis']
        
        data = {}
        
        # Process DQ signals
        for i, dq_port in enumerate(self.dq_left_ports):
            signal_name = f"DQ{i}"
            port_key = f"Port_{dq_port}"
            right_port_key = f"Port_{self.dq_right_ports[i]}"
            
            data[signal_name] = {
                "tdr_left": tdr_all_ports[port_key].tolist(),
                "tdr_right": tdr_all_ports[right_port_key].tolist(),
                "time_axis_ns": (time_axis * 1e9).tolist()
            }
        
        # Process DQS signals (differential TDR)
        for dqs_name in self.dqs_left_ports:
            left_port_key = f"Port_{self.dqs_left_ports[dqs_name]}"
            right_port_key = f"Port_{self.dqs_right_ports[dqs_name]}"
            
            # For differential signals, we might want to calculate differential TDR
            # For now, using single-ended TDR
            data[dqs_name] = {
                "tdr_left": tdr_all_ports[left_port_key].tolist(),
                "tdr_right": tdr_all_ports[right_port_key].tolist(),
                "time_axis_ns": (time_axis * 1e9).tolist()
            }
        
        return data
    
    def process_all_models(self, processor, model_names: List[str]) -> Tuple[Dict, Dict, Dict]:
        """
        Process all models and return organized data for collectors.
        
        Args:
            processor: DDRTouchstoneProcessor instance
            model_names: List of model names
            
        Returns:
            Tuple of (s_matrix_data, metrics_data, tdr_data) dictionaries
        """
        s_matrix_data = {}
        metrics_data = {}
        tdr_data = {}
        
        for model_name in model_names:
            s_params = processor.models[model_name]['s_params']
            frequencies = processor.models[model_name]['frequencies']
            
            # Process each type of data
            s_matrix_data[model_name] = self.process_s_matrix_data(s_params, frequencies, model_name)
            metrics_data[model_name] = self.process_ddr_metrics(s_params, frequencies, model_name)
            tdr_data[model_name] = self.process_tdr_data(s_params, frequencies, model_name)
        
        return s_matrix_data, metrics_data, tdr_data 