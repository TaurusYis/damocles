"""
data_organizer.py
Module for organizing processed S-parameter data into a single HierarchicalDataCollector instance.
"""
from typing import Dict, Any
from data_collector import HierarchicalDataCollector

class DataOrganizer:
    """
    Organizes processed S-parameter data into a single HierarchicalDataCollector instance
    with a comprehensive hierarchy: [model, data_type, identifier]
    """
    
    @staticmethod
    def organize_all_data(s_matrix_data: Dict, metrics_data: Dict, tdr_data: Dict) -> HierarchicalDataCollector:
        """
        Organize all processed data into a single HierarchicalDataCollector.
        
        Hierarchy: [model, data_type, identifier]
        - model: model name (e.g., 'CPU_Socket_v1')
        - data_type: 's_matrix', 'metrics', or 'tdr'
        - identifier: 
          - for s_matrix: port_pair (e.g., '(1,1)', '(1,2)', etc.)
          - for metrics: signal name (e.g., 'DQ0', 'DQS0-', etc.)
          - for tdr: signal name (e.g., 'DQ0', 'DQS0-', etc.)
        
        Args:
            s_matrix_data: Processed S-matrix data
            metrics_data: Processed metrics data
            tdr_data: Processed TDR data
            
        Returns:
            Single HierarchicalDataCollector with all data organized
        """
        collector = HierarchicalDataCollector(["model", "data_type", "identifier"])
        
        # Organize S-matrix data
        for model_name, model_data in s_matrix_data.items():
            for port_pair, metrics in model_data.items():
                for metric_key, metric_value in metrics.items():
                    collector.add(
                        model_name, "s_matrix", port_pair,
                        metric_key=metric_key, metric_value=metric_value
                    )
        
        # Organize metrics data
        for model_name, model_data in metrics_data.items():
            for signal_name, metrics in model_data.items():
                for metric_key, metric_value in metrics.items():
                    collector.add(
                        model_name, "metrics", signal_name,
                        metric_key=metric_key, metric_value=metric_value
                    )
        
        # Organize TDR data
        for model_name, model_data in tdr_data.items():
            for signal_name, metrics in model_data.items():
                for metric_key, metric_value in metrics.items():
                    collector.add(
                        model_name, "tdr", signal_name,
                        metric_key=metric_key, metric_value=metric_value
                    )
        
        return collector
    
    @staticmethod
    def extract_s_matrix_data(collector: HierarchicalDataCollector) -> Dict:
        """
        Extract S-matrix data from the unified collector.
        
        Args:
            collector: Unified HierarchicalDataCollector
            
        Returns:
            Dictionary with S-matrix data organized by model and port_pair
        """
        data = collector.get()
        s_matrix_data = {}
        
        for model_name in data:
            if "s_matrix" in data[model_name]:
                s_matrix_data[model_name] = data[model_name]["s_matrix"]
        
        return s_matrix_data
    
    @staticmethod
    def extract_metrics_data(collector: HierarchicalDataCollector) -> Dict:
        """
        Extract metrics data from the unified collector.
        
        Args:
            collector: Unified HierarchicalDataCollector
            
        Returns:
            Dictionary with metrics data organized by model and signal
        """
        data = collector.get()
        metrics_data = {}
        
        for model_name in data:
            if "metrics" in data[model_name]:
                metrics_data[model_name] = data[model_name]["metrics"]
        
        return metrics_data
    
    @staticmethod
    def extract_tdr_data(collector: HierarchicalDataCollector) -> Dict:
        """
        Extract TDR data from the unified collector.
        
        Args:
            collector: Unified HierarchicalDataCollector
            
        Returns:
            Dictionary with TDR data organized by model and signal
        """
        data = collector.get()
        tdr_data = {}
        
        for model_name in data:
            if "tdr" in data[model_name]:
                tdr_data[model_name] = data[model_name]["tdr"]
        
        return tdr_data