import pickle
import pandas as pd
from pathlib import Path
from collections import defaultdict

class HierarchicalDataCollector:
    def __init__(self, levels):
        """
        levels: List of strings representing each hierarchy level.
        e.g., ["system", "device", "channel"]
        """
        self.levels = levels
        self.data = self._create_nested_dict(len(levels))

    def _create_nested_dict(self, depth):
        if depth == 1:
            return defaultdict(dict)
        return defaultdict(lambda: self._create_nested_dict(depth - 1))

    def _get_node(self, *keys):
        """Navigate to the node just before the final metric dict."""
        node = self.data
        for key in keys[:-1]:
            node = node[key]
        return node

    # ---- Optional Structure Setup ----
    def define_metrics(self, *args, metric_keys):
        """
        Initialize all metric keys with None or a placeholder for a given path.
        """
        assert len(args) == len(self.levels), "Mismatch in hierarchy depth"
        node = self._get_node(*args)
        node[args[-1]] = defaultdict(lambda: None, {key: None for key in metric_keys})

    def set_metric(self, *args, metric_key, metric_value):
        """
        Set or update a signle metric value, assuming the metric key already exists.
        """
        assert len(args) == len(self.levels), "Mismatch in hierarchy depth"
        node = self._get_node(*args)
        if args[-1] not in node:
            node[args[-1]] = defaultdict(lambda: None)
        node[args[-1]][metric_key] = metric_value

    # ---- Add / Update Data ----
    def add(self, *args, metric_key, metric_value):
        """
        Add a single metric.
        args: list of keys for each level, ending before metrics dict
        """
        assert len(args) == len(self.levels), "Mismatch in hierarchy depth"
        node = self._get_node(*args)
        node[args[-1]][metric_key] = metric_value

    def update(self, *args, metrics: dict):
        """
        Add/update multiple metrics.
        args: list of keys for each level, ending before metrics dict
        """
        assert len(args) == len(self.levels), "Mismatch in hierarchy depth"
        node = self._get_node(*args)
        node[args[-1]].update(metrics)

    # ---- Save / Load Data ----
    def save_pickle(self, file_path):
        # Convert defaultdict to regular dict for pickling
        def convert_defaultdict_to_dict(d):
            if isinstance(d, defaultdict):
                return {k: convert_defaultdict_to_dict(v) for k, v in d.items()}
            return d
        
        with open(file_path, 'wb') as f:
            pickle.dump(convert_defaultdict_to_dict(self.data), f)

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            raw = pickle.load(f)
            self.data = self._convert_to_nested_defaultdict(raw)

    def _convert_to_nested_defaultdict(self, d, depth=0):
        if depth >= len(self.levels) - 1:
            return defaultdict(dict, d)
        return defaultdict(lambda: self._create_nested_dict(len(self.levels) - depth - 1),
                           {k: self._convert_to_nested_defaultdict(v, depth + 1) for k, v in d.items()})

    # ---- Export Data ----
    def export_to_excel(self, excel_path, flatten_lists=True):
        rows = []

        def walk(level_dict, keys_so_far):
            if len(keys_so_far) == len(self.levels):
                row = dict(zip(self.levels, keys_so_far))
                for metric_key, metric_value in level_dict.items():
                    if flatten_lists and isinstance(metric_value, list):
                        metric_value = ', '.join(map(str, metric_value))
                    row[metric_key] = metric_value
                rows.append(row)
            else:
                for key, subdict in level_dict.items():
                    walk(subdict, keys_so_far + [key])

        walk(self.data, [])
        df = pd.DataFrame(rows)
        Path(excel_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(excel_path, index=False)

    def get(self):
        return self.data






