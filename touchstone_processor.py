"""
touchstone_processor.py
Module for reading and processing Touchstone files, and calculating DDR metrics.
"""
import numpy as np
import os
import warnings

try:
    import skrf as rf
    SKRF_AVAILABLE = True
except ImportError:
    SKRF_AVAILABLE = False
    warnings.warn("scikit-rf not available, using manual Touchstone parsing.")

class DDRTouchstoneProcessor:
    def __init__(self):
        self.models = {}
        self.frequencies = None
        self.port_names = []

    def read_touchstone_file(self, filepath: str, model_name: str) -> bool:
        try:
            if SKRF_AVAILABLE:
                network = rf.Network(filepath)
                self.models[model_name] = {
                    'frequencies': network.f,
                    's_params': network.s,
                    'port_names': [f'Port_{i+1}' for i in range(network.nports)]
                }
            else:
                self._parse_touchstone_manual(filepath, model_name)
            if self.frequencies is None:
                self.frequencies = self.models[model_name]['frequencies']
                self.port_names = self.models[model_name]['port_names']
            return True
        except Exception as e:
            print(f"Error reading {filepath}: {str(e)}")
            return False

    def _parse_touchstone_manual(self, filepath: str, model_name: str):
        with open(filepath, 'r') as f:
            lines = f.readlines()
        header_line = lines[0].strip()
        freq_unit = 'GHz' if 'GHz' in header_line else 'Hz'
        format_type = 'RI' if 'RI' in header_line else 'MA'
        data_lines = []
        for line in lines[1:]:
            line = line.strip()
            if line and not line.startswith('!'):
                data_lines.append(line)
        data = np.array([[float(x) for x in line.split()] for line in data_lines])
        frequencies = data[:, 0]
        if freq_unit == 'GHz':
            frequencies *= 1e9
        s_params = data[:, 1:]
        if format_type == 'RI':
            s_params_complex = s_params[:, 0::2] + 1j * s_params[:, 1::2]
        else:
            mag = s_params[:, 0::2]
            phase = s_params[:, 1::2] * np.pi / 180
            s_params_complex = mag * np.exp(1j * phase)
        nports = int(np.sqrt(s_params_complex.shape[1]))
        s_params_reshaped = s_params_complex.reshape(-1, nports, nports)
        self.models[model_name] = {
            'frequencies': frequencies,
            's_params': s_params_reshaped,
            'port_names': [f'Port_{i+1}' for i in range(nports)]
        }

    def define_ddr_ports(self, dq_ports, dqs_ports, dqs_pairs):
        self.dq_ports = dq_ports
        self.dqs_ports = dqs_ports
        self.dqs_pairs = dqs_pairs

    def calculate_insertion_loss(self, model_name: str):
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        il_results = {}
        for dq_port in self.dq_ports:
            il_db = 20 * np.log10(np.abs(s_params[:, dq_port-1, dq_port-1]))
            il_results[f'DQ_{dq_port}'] = il_db
        for i, (pos_port, neg_port) in enumerate(self.dqs_pairs):
            s_pos = s_params[:, pos_port-1, pos_port-1]
            s_neg = s_params[:, neg_port-1, neg_port-1]
            s_diff = s_pos - s_neg
            il_db = 20 * np.log10(np.abs(s_diff))
            il_results[f'DQS_{i+1}_diff'] = il_db
        return il_results

    def calculate_return_loss(self, model_name: str):
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        rl_results = {}
        for i in range(s_params.shape[1]):
            rl_db = 20 * np.log10(np.abs(s_params[:, i, i]))
            rl_results[f'Port_{i+1}'] = rl_db
        return rl_results

    def calculate_tdr(self, model_name: str, time_window: float = 10e-9):
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        frequencies = model_data['frequencies']
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

    def calculate_crosstalk(self, model_name: str):
        model_data = self.models[model_name]
        s_params = model_data['s_params']
        crosstalk_results = {}
        for i, dq1 in enumerate(self.dq_ports):
            for j, dq2 in enumerate(self.dq_ports):
                if i != j:
                    fext_db = 20 * np.log10(np.abs(s_params[:, dq2-1, dq1-1]))
                    crosstalk_results[f'FEXT_DQ{dq1}_to_DQ{dq2}'] = fext_db
        for i, dq1 in enumerate(self.dq_ports):
            for j, dq2 in enumerate(self.dq_ports):
                if i != j:
                    next_db = 20 * np.log10(np.abs(s_params[:, dq1-1, dq2-1]))
                    crosstalk_results[f'NEXT_DQ{dq1}_to_DQ{dq2}'] = next_db
        return crosstalk_results 