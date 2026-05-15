"""
Load UCI Gas Sensor Array Drift dataset and split into two virtual devices (batches 1-5 vs 6-10).
Returns:
    X_source, y_source: readings and gas labels for source device (batches 1-5)
    X_target, y_target: readings and gas labels for target device (batches 6-10)
All readings are raw (not log-transformed) and shaped (n_samples, 128).
"""
import numpy as np
import os
from glob import glob

def load_uci_data(data_dir="data/uci/gas+sensor+array+drift+dataset/Dataset"):
    X_source, y_source = [], []
    X_target, y_target = [], []
    for filepath in sorted(glob(os.path.join(data_dir, "batch*.dat"))):
        batch_num = int(filepath.split('batch')[1].split('.')[0])
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[0])
                features = [float(val.split(':')[1]) for val in parts[1:129]]
                if len(features) != 128:
                    continue
                if batch_num <= 5:
                    X_source.append(features)
                    y_source.append(label)
                else:
                    X_target.append(features)
                    y_target.append(label)
    X_source = np.array(X_source)
    X_target = np.array(X_target)
    y_source = np.array(y_source)
    y_target = np.array(y_target)
    print(f"Source device (batches 1-5): {len(X_source)} samples")
    print(f"Target device (batches 6-10): {len(X_target)} samples")
    return X_source, y_source, X_target, y_target